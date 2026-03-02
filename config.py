import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
OPENNINJA_API_KEY = os.getenv("OPENNINJA_API_KEY") # Deprecated for this project

# Services
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Embedding Config
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # Default for text-embedding-3-small

# Agent Models
WORKER_MODEL = "gpt-4o-mini"
ORCHESTRATOR_MODEL = "gpt-4o"
