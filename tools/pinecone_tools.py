from memory.vector_store import PineconeVectorStore
from typing import Optional

# Initialize a global instance so we don't recreate clients constantly
store = PineconeVectorStore()

async def store_to_pinecone(chunk_id: str, text: str, source_url: str = "", metadata: Optional[dict] = None) -> str:
    """
    Store a text chunk into the Pinecone RAG memory.
    """
    meta = metadata or {}
    if source_url:
        meta["source_url"] = source_url
        
    try:
        await store.store_chunk(chunk_id, text, meta)
        return f"Successfully stored chunk '{chunk_id}' to Pinecone."
    except Exception as e:
        return f"Failed to store chunk '{chunk_id}' to Pinecone: {e}"

async def rag_query_pinecone(query: str, top_k: int = 5) -> str:
    """
    Query the Pinecone RAG memory to retrieve relevant context chunks.
    """
    try:
        results = await store.search(query, top_k=top_k)
        
        if not results:
            return "No relevant context found in memory."
            
        contexts = []
        for match in results:
            meta = match.get("metadata", {})
            text = meta.get("text", "")
            source = meta.get("source_url", "Unknown Source")
            score = match.get("score", 0.0)
            contexts.append(f"Source: {source} (Score: {score:.2f})\nText: {text}")
            
        return "\n\n---\n\n".join(contexts)
    except Exception as e:
        return f"Error querying Pinecone: {e}"
