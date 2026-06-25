"""
Embeddings Utility
Generates vector embeddings using Google Gemini embedding model
and manages the ChromaDB vector store.
"""

import os
import logging
import uuid
from typing import List, Dict, Any, Optional

import chromadb
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Gemini embedding model (gemini-embedding-001 is current stable)
EMBEDDING_MODEL = "models/gemini-embedding-001"
COLLECTION_NAME = "research_documents"


def get_chroma_client(persist_dir: str = "./vectorstore") -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the documents collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts_batch(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """
    Batch-embed multiple texts in a single API call (much faster than one-by-one).
    Gemini embedding-004 supports batching via embed_content with a list.
    Falls back to one-by-one if batch fails.
    """
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texts,
            task_type=task_type,
        )
        # Batch result returns list of embeddings
        if isinstance(result["embedding"][0], list):
            return result["embedding"]
        # Single-item batch returns flat list
        return [result["embedding"]]
    except Exception:
        # Fallback: embed one-by-one
        embeddings = []
        for text in texts:
            r = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type=task_type,
            )
            embeddings.append(r["embedding"])
        return embeddings


def embed_query(query: str) -> List[float]:
    """Generate embedding for a search query."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY",
    )
    return result["embedding"]


def store_chunks(
    chunks: List[Dict[str, Any]],
    collection: chromadb.Collection,
    source_name: str,
    batch_size: int = 50,
) -> int:
    """
    Embed and store chunks in ChromaDB using batched embedding calls.
    Processes in batches of `batch_size` to avoid API limits.
    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    total_stored = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]

        try:
            embeddings = embed_texts_batch(texts, task_type="RETRIEVAL_DOCUMENT")
        except Exception as e:
            logger.error(f"Embedding batch {i // batch_size} failed: {e}")
            raise

        documents = texts
        metadatas = [
            {
                "source": c["source"],
                "page": int(c["page"]),
                "chunk_index": int(c["chunk_index"]),
            }
            for c in batch
        ]
        ids = [str(uuid.uuid4()) for _ in batch]

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        total_stored += len(batch)
        logger.info(f"Stored batch {i // batch_size + 1}: {len(batch)} chunks for '{source_name}'")

    return total_stored


def retrieve_relevant_chunks(
    query: str,
    collection: chromadb.Collection,
    n_results: int = 5,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search: find the top-n most relevant chunks for a query.
    Guards against n_results exceeding collection size.
    """
    query_embedding = embed_query(query)

    # Guard: n_results cannot exceed total docs in collection
    total = collection.count()
    if total == 0:
        return []
    n_results = min(n_results, total)

    where_filter = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results and results.get("documents") and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", "?"),
                "relevance_score": round(1 - float(dist), 3),
            })

    return chunks


def delete_source_documents(collection: chromadb.Collection, source_name: str) -> None:
    """Remove all chunks belonging to a specific source file."""
    try:
        collection.delete(where={"source": source_name})
        logger.info(f"Deleted chunks for '{source_name}' from ChromaDB")
    except Exception as e:
        # If no docs matched, chromadb may raise — log and continue
        logger.warning(f"Delete warning for '{source_name}': {e}")


def get_collection_stats(collection: chromadb.Collection) -> Dict[str, Any]:
    """Return basic stats about the collection."""
    count = collection.count()
    return {"total_chunks": count}
