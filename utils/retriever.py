"""
Retriever Utility
High-level interface for retrieving context chunks from ChromaDB
and formatting them for use in prompts.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.
    """
    if not chunks:
        return ""

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def format_sources_for_display(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format chunks into a clean list for displaying to the user as citations.
    Deduplicates by (source, page).
    """
    seen = set()
    sources = []

    for i, chunk in enumerate(chunks, 1):
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "index": len(sources) + 1,
                "source": chunk["source"],
                "page": chunk["page"],
                "relevance": chunk.get("relevance_score", 0),
                "snippet": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
            })

    return sources


def build_qa_prompt(question: str, context: str) -> str:
    """Build the full QA prompt with context."""
    return f"""You are an expert research assistant. Use the provided document context to answer the question accurately and thoroughly.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer based on the provided context only
- Be specific and cite which source/page your answer comes from when relevant
- If the context doesn't contain enough information to answer, say so clearly
- Format your answer clearly with proper structure (use bullet points or numbered lists where appropriate)
- If the question asks for comparisons or analysis, provide structured insights

ANSWER:"""


def build_summary_prompt(context: str, doc_names: List[str]) -> str:
    """Build the document summarization prompt."""
    docs_list = ", ".join(doc_names)
    return f"""You are an expert research analyst. Analyze the following document content and provide a comprehensive structured summary.

DOCUMENTS: {docs_list}

CONTENT:
{context}

Generate a detailed summary with these exact sections:

## Executive Summary
(2-3 sentence overview of what this document is about and its main purpose)

## Key Topics
(Bullet points of the main subjects, themes, or areas covered)

## Important Findings
(The most significant facts, results, conclusions, or arguments presented)

## Methodology (if applicable)
(How the research/analysis was conducted, if relevant)

## Conclusion
(The main takeaways and implications)

Be thorough, accurate, and base everything strictly on the provided content."""
