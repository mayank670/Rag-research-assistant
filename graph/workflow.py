"""
LangGraph RAG Workflow
Pipeline: User Question → Retrieve → Validate → Generate → Respond
"""

import logging
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
import google.generativeai as genai

from utils.embeddings import retrieve_relevant_chunks
from utils.retriever import format_context_for_prompt, format_sources_for_display, build_qa_prompt

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    collection: Any
    retrieved_chunks: List[Dict[str, Any]]
    context: str
    has_context: bool
    answer: str
    sources: List[Dict[str, Any]]
    error: Optional[str]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def retrieval_node(state: RAGState) -> dict:
    logger.info(f"[Retrieval] Query: {state['question'][:80]}")
    try:
        chunks = retrieve_relevant_chunks(
            query=state["question"],
            collection=state["collection"],
            n_results=5,
        )
        logger.info(f"[Retrieval] Got {len(chunks)} chunks")
        return {"retrieved_chunks": chunks, "error": None}
    except Exception as e:
        logger.error(f"[Retrieval] {e}")
        return {"retrieved_chunks": [], "error": f"Retrieval failed: {str(e)}"}


def validation_node(state: RAGState) -> dict:
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        logger.warning("[Validation] No chunks retrieved")
        return {"has_context": False, "context": "", "sources": []}

    # Lower threshold to 0.2 to be more inclusive
    relevant = [c for c in chunks if c.get("relevance_score", 1) >= 0.2]

    if not relevant:
        logger.warning("[Validation] All chunks below threshold — using top chunk anyway")
        relevant = chunks[:1]  # use best chunk rather than failing

    context = format_context_for_prompt(relevant)
    sources = format_sources_for_display(relevant)
    logger.info(f"[Validation] {len(relevant)} relevant chunks, {len(context)} chars")
    return {
        "has_context": True,
        "context": context,
        "sources": sources,
        "retrieved_chunks": relevant,
    }


def generation_node(state: RAGState) -> dict:
    if not state.get("has_context"):
        return {
            "answer": (
                "I couldn't find relevant information in the uploaded documents. "
                "Try rephrasing your question or uploading a different document."
            )
        }

    logger.info("[Generation] Calling Gemini...")
    try:
        prompt = build_qa_prompt(state["question"], state["context"])
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )
        answer = response.text
        logger.info(f"[Generation] {len(answer)} chars generated")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"[Generation] {e}")
        return {"answer": f"Generation failed: {str(e)}", "error": str(e)}


def response_node(state: RAGState) -> dict:
    logger.info("[Response] Done")
    return {
        "answer": state.get("answer", "No answer generated."),
        "sources": state.get("sources", []),
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_validation(state: RAGState) -> str:
    if state.get("error"):
        return "response"
    return "generation"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("validation", validation_node)
    graph.add_node("generation", generation_node)
    graph.add_node("response", response_node)
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "validation")
    graph.add_conditional_edges(
        "validation",
        route_after_validation,
        {"generation": "generation", "response": "response"},
    )
    graph.add_edge("generation", "response")
    graph.add_edge("response", END)
    return graph.compile()


def run_rag_pipeline(question: str, collection) -> Dict[str, Any]:
    """Run the full RAG pipeline. Returns answer, sources, error, chunks_used."""
    graph = build_rag_graph()

    initial: RAGState = {
        "question": question,
        "collection": collection,
        "retrieved_chunks": [],
        "context": "",
        "has_context": False,
        "answer": "",
        "sources": [],
        "error": None,
    }

    try:
        final = graph.invoke(initial)
        return {
            "answer": final["answer"],
            "sources": final["sources"],
            "error": final.get("error"),
            "chunks_used": len(final.get("retrieved_chunks", [])),
        }
    except Exception as e:
        logger.error(f"[Pipeline] {e}")
        return {
            "answer": f"Pipeline error: {str(e)}",
            "sources": [],
            "error": str(e),
            "chunks_used": 0,
        }
