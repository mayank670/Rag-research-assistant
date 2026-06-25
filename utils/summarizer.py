"""
Summarizer Utility
Handles document-level summarization using Gemini.
"""

import logging
from typing import List, Dict, Any
import google.generativeai as genai

from utils.retriever import build_summary_prompt

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


def generate_summary(
    collection,
    doc_names: List[str],
    max_chunks: int = 20,
) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of uploaded documents.
    Samples chunks from ChromaDB to build representative context.
    """
    if not doc_names:
        return {"error": "No documents available to summarize."}

    try:
        total = collection.count()
        if total == 0:
            return {"error": "No content found. Please re-upload your documents."}

        # collection.get() with include — limit not supported in all versions
        # Use peek() which is always available, then get() without limit
        try:
            results = collection.get(
                limit=max_chunks,
                include=["documents", "metadatas"],
            )
        except TypeError:
            # Older chromadb: .get() doesn't support limit keyword
            results = collection.peek(limit=max_chunks)

        if not results or not results.get("documents"):
            return {"error": "Could not retrieve document content from the store."}

        context_parts = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            context_parts.append(
                f"[{meta.get('source', 'Unknown')}, p.{meta.get('page', '?')}]\n{doc}"
            )

        context = "\n\n".join(context_parts)
        prompt = build_summary_prompt(context, doc_names)

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )

        return {
            "summary": response.text,
            "documents": doc_names,
            "chunks_used": len(results["documents"]),
        }

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return {"error": f"Summary failed: {str(e)}"}
