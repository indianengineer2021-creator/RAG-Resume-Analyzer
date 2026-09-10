import json
import logging
from typing import Any, Dict

from src.llm import get_llm
from src.prompts import build_follow_up_prompt, build_resume_analysis_prompt
from src.retriever import retrieve_relevant_documents
from src.vector_store import create_vector_store

logger = logging.getLogger(__name__)


def _safe_json_loads(raw_response: str) -> Dict[str, Any]:
    """Parse the model output and fall back gracefully if JSON structure is not perfect."""
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("` ")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            return json.loads(candidate)
        raise ValueError("Gemini response could not be parsed as JSON.")


def analyze_resume(job_description: str, vector_store=None, k: int = 5) -> Dict[str, Any]:
    """Run the full RAG pipeline for resume-to-job analysis using the retrieved resume context."""
    if not job_description or not job_description.strip():
        raise ValueError("A valid Job Description is required.")

    if vector_store is None:
        vector_store = create_vector_store()

    logger.info("Retrieving resume chunks for the job description.")
    relevant_chunks = retrieve_relevant_documents(vector_store=vector_store, query=job_description, k=k)
    context = "\n\n---\n\n".join(chunk.page_content for chunk in relevant_chunks)

    if not context.strip():
        raise ValueError("No resume context was retrieved for the provided job description.")

    prompt = build_resume_analysis_prompt(job_description=job_description, context=context)
    logger.info("Sending retrieved context to Gemini for grounded resume analysis.")
    llm = get_llm()
    response = llm.invoke(prompt)
    response_text = getattr(response, "content", str(response))
    return _safe_json_loads(response_text)


def answer_question(question: str, vector_store=None, k: int = 5) -> str:
    """Answer a follow-up question using the same RAG retriever and a grounded prompt."""
    if not question or not question.strip():
        raise ValueError("A valid question is required.")

    if vector_store is None:
        vector_store = create_vector_store()

    relevant_chunks = retrieve_relevant_documents(vector_store=vector_store, query=question, k=k)
    context = "\n\n---\n\n".join(chunk.page_content for chunk in relevant_chunks)
    if not context.strip():
        return "Not found in resume"

    prompt = build_follow_up_prompt(question, context)
    llm = get_llm()
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))
