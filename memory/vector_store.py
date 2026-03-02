import os
import sys
import time
import asyncio
from typing import List, Dict, Optional

from pinecone import Pinecone, ServerlessSpec
from openai import AsyncOpenAI

# Determine the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

# Initialize clients
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    print(f"Warning: Failed to initialize Pinecone: {e}")
    pc = None

try:
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Warning: Failed to initialize OpenAI: {e}")
    openai_client = None


class PineconeVectorStore:
    def __init__(self, index_name: str = "research-engine"):
        self.index_name = index_name
        self.index = None
        if pc:
            self.initialize_index()

    def initialize_index(self):
        """Creates the index if it doesn't exist."""
        try:
            existing_indexes = [index.name for index in pc.list_indexes()]
            if self.index_name not in existing_indexes:
                print(f"Creating Pinecone index '{self.index_name}'...")
                pc.create_index(
                    name=self.index_name,
                    dimension=EMBEDDING_DIMENSIONS,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=PINECONE_ENVIRONMENT
                    )
                )
                # Wait for index to be ready
                while not pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
            
            self.index = pc.Index(self.index_name)
        except Exception as e:
            print(f"Error initializing Pinecone index: {e}")

    async def get_embedding(self, text: str) -> List[float]:
        """Fetches the embedding for a given text using OpenAI."""
        if not openai_client:
            raise ValueError("OpenAI client is not initialized.")
            
        # Optional: Handle potentially long text by truncating or warning
        response = await openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding

    async def store_chunk(self, chunk_id: str, text: str, metadata: dict = None):
        """Stores a single chunk with its text and metadata."""
        if not self.index:
            print("Pinecone index not available. Skipping store.")
            return

        embedding = await self.get_embedding(text)
        
        meta = metadata or {}
        meta["text"] = text  # Store the raw text to retrieve it later
        
        # Async run in executor since pinecone client is sync
        await asyncio.to_thread(
            self.index.upsert,
            vectors=[{"id": chunk_id, "values": embedding, "metadata": meta}]
        )

    async def store_chunks_batch(self, chunks: List[Dict]):
        """Stores multiple chunks. Expected format: {"id": str, "text": str, "metadata": dict}"""
        if not self.index:
            print("Pinecone index not available. Skipping store.")
            return

        vectors = []
        for chunk in chunks:
            try:
                embedding = await self.get_embedding(chunk["text"])
                meta = chunk.get("metadata", {})
                meta["text"] = chunk["text"]
                vectors.append({"id": chunk["id"], "values": embedding, "metadata": meta})
            except Exception as e:
                print(f"Error embedding chunk {chunk.get('id')}: {e}")

        if vectors:
            await asyncio.to_thread(self.index.upsert, vectors=vectors)

    async def search(self, query: str, top_k: int = 5, filter_dict: dict = None) -> List[Dict]:
        """Searches the vector store using cosine similarity."""
        if not self.index:
            print("Pinecone index not available.")
            return []

        query_embedding = await self.get_embedding(query)
        
        kwargs = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        }
        if filter_dict:
            kwargs["filter"] = filter_dict
            
        response = await asyncio.to_thread(lambda: self.index.query(**kwargs))
        return response.matches
