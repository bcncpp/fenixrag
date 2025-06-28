from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List
import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.settings import settings

logger = logging.getLogger(__name__)


class GeminiEmbeddings(Embeddings):
    """Gemini embeddings with error handling and retries"""
    
    def __init__(self):
        """Initialize Gemini embeddings"""
        
        if not settings.validate_api_key():
            raise ValueError(
                "Invalid Google API key. Please set GOOGLE_API_KEY in your .env file. "
                "Get your key from: https://makersuite.google.com/app/apikey"
            )
        
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key
        )
        
        logger.info(f"✅ Initialized Gemini embeddings with model: {settings.gemini_model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with retry logic"""
        
        try:
            # Process in batches to avoid rate limits
            batch_size = min(settings.max_batch_size, 50)  # Gemini has rate limits
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                batch_embeddings = self._embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Small delay between batches to respect rate limits
                if i + batch_size < len(texts):
                    asyncio.sleep(0.1)
            
            logger.info(f"✅ Generated embeddings for {len(texts)} documents")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating document embeddings: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with retry logic"""
        
        try:
            embedding = self._embeddings.embed_query(text)
            logger.debug(f"✅ Generated query embedding for: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generating query embedding: {e}")
            raise
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents"""
        
        # Run in thread pool since Gemini client is sync
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query"""
        
        # Run in thread pool since Gemini client is sync
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)
