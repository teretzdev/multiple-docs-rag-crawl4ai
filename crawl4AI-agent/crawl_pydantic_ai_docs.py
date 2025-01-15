can just use the saimport os
import sys
import json
import asyncio
import requests
import time
import random
import logging
import argparse
from xml.etree import ElementTree
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from gemi_flash import FlashModel  # Assuming gemi_flash is the module name
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text(text: str, max_tokens: int = 4000) -> List[str]:
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for paragraph in text.split("\\\\\\\n\\\\\\\n"):
        paragraph_tokens = num_tokens_from_string(paragraph)
        if current_tokens + paragraph_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
            current_tokens = paragraph_tokens
        else:
            current_chunk += "\\\\\\\n\\\\\\\n" + paragraph
            current_tokens += paragraph_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def async_retry(coroutine):
    return await coroutine

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using the specified model."""
    use_flash_model = os.getenv("USE_FLASH_MODEL", "false").lower() == "true"
    
    if use_flash_model:
        flash_model = FlashModel(api_key=os.getenv("GEMI_FLASH_API_KEY"))
        return flash_model.extract_title_and_summary(chunk, url)
    else:
        system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
        For the summary: Create a concise summary of the main points in this chunk.
        Keep both title and summary concise but informative."""
        
        try:
            response = await async_retry(openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"URL: {url}\\\\\\\n\\\\\\\nContent:\\\\\\\n{chunk[:1000]}..."}  # Send first 1000 chars for context
                ],
                response_format={ "type": "json_object" }
            ))
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in API call: {e}", exc_info=True)
            return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await async_retry(openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ))
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error in API call: {e}", exc_info=True)
        return [0] * 1536  # Return zero vector on error

rate_limit = AsyncLimiter(max_rate=50, time_period=60)  # 50 requests per minute

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    async with rate_limit:
        await asyncio.sleep(1)  # Add a 1-second delay before API calls
        # Get title and summary
        extracted = await get_title_and_summary(chunk, url)
        
        # Get embedding
        embedding = await get_embedding(chunk)
        
        # Create metadata
        metadata = {
            "source": "pydantic_ai_docs",
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path
        }
        
        return ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=extracted['title'],
            summary=extracted['summary'],
            content=chunk,  # Store the original chunk content
            metadata=metadata,
            embedding=embedding
        )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

def get_file_size(url: str) -> int:
    """Get the size of the file at the given URL."""
    response = requests.head(url)
    return int(response.headers.get('content-length', 0))

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    current_file_size = get_file_size(url)
    local_data = load_vector_data_offline(url)
    
    if local_data and local_data['file_size'] == current_file_size:
        print(f"Loading vector data from local storage for {url}")
        return local_data['chunks']
    
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)
    
    # Save vector data offline
    save_vector_data_offline(url, processed_chunks, current_file_size)
    
    # Save vector data offline
    save_vector_data_offline(processed_chunks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_pydantic_ai_docs_urls() -> List[str]:
    """Get URLs from Pydantic AI docs sitemap."""
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

def save_vector_data_offline(url: str, chunks: List[ProcessedChunk], file_size: int, filename: str = 'vector_data.json'):
    """Save processed chunks and their embeddings to a local file."""
    data = {
        'url': url,
        'file_size': file_size,
        'chunks': [chunk.__dict__ for chunk in chunks]
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_vector_data_offline(url: str, filename: str = 'vector_data.json') -> Optional[Dict[str, Any]]:
    """Load vector data from a local file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        if data['url'] == url:
            return data
    except FileNotFoundError:
        return None
    return None

def get_text_content(input_text: str) -> str:
    """Get text content from a URL or local file path."""
    try:
        result = urlparse(input_text)
        if all([result.scheme, result.netloc]):
            # It's a URL
            return requests.get(input_text).text
        elif os.path.isfile(input_text):
            # It's a local file
            with open(input_text, 'r') as f:
                return f.read()
    except Exception as e:
        print(f"Error fetching content: {e}")
    # Return the input text as is if it's neither a URL nor a file
    return input_text

async def add_manual_text(text: str):
    """Add text manually for embedding and save it."""
    # Fetch the content
    content = get_text_content(text)
    # Process the content
    chunk = await process_chunk(content, 0, "manual_input")
    
    # Insert into Supabase
    await insert_chunk(chunk)
    
    # Save offline
    save_vector_data_offline([chunk])

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--crawl', action='store_true', help='Initiate the crawling process')
    parser.add_argument('--add-text', type=str, help='Add manual text for embedding')
    parser.add_argument('--crud', action='store_true', help='Perform CRUD operations on vector data')
    parser.add_argument('--use-flash-model', action='store_true', help="Use Gemi's Flash model")
    return parser.parse_args()

async def main(args):
    if args.crawl:
        urls = get_pydantic_ai_docs_urls()
        if not urls:
            print("No URLs found to crawl")
            return
        print(f"Found {len(urls)} URLs to crawl")
        await crawl_parallel(urls)
    elif args.add_text:
        text_input = args.add_text
        await add_manual_text(text_input)
    elif args.crud:
        print("CRUD operations are not yet implemented.")
    elif args.use_flash_model:
        os.environ["USE_FLASH_MODEL"] = "true"
        print("Using Gemi's Flash Model for processing.")
    else:
        print("No valid operation specified.")

if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(args))