import json
import logging
import re
from typing import Any, Dict

from src.llm import get_llm
from src.prompts import build_follow_up_prompt, build_resume_analysis_prompt
from src.retriever import retrieve_relevant_documents
from src.vector_store import create_vector_store

logger = logging.getLogger(__name__)


def _response_to_text(response: Any) -> str:
    """Normalize Gemini string or structured content-block responses to text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    text_parts.append(_response_to_text(value))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts)
    if isinstance(content, dict):
        value = content.get("text") or content.get("content")
        return _response_to_text(value) if value else str(content)
    return str(content)


def _fallback_analysis(raw_response: str) -> Dict[str, Any]:
    """Keep a non-JSON Gemini answer visible instead of failing the whole analysis."""
    return {
        "overall_score": 0,
        "summary": raw_response.strip() or "Gemini returned an empty analysis.",
        "skill_match": [],
        "missing_skills": [],
        "experience_match": "Not found in resume",
        "domain_match": "Not found in resume",
        "certification_match": "Not found in resume",
        "strengths": [],
        "weaknesses": [],
        "ats_gaps": [],
        "recommendations": [],
        "interview_topics": [],
    }


def _safe_json_loads(raw_response: str) -> Dict[str, Any]:
    """Parse the model output and fall back gracefully if JSON structure is not perfect."""
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else _fallback_analysis(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for start, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[start:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        logger.warning("Gemini response was not valid JSON; returning text fallback.")
        return _fallback_analysis(cleaned)


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
    return _safe_json_loads(_response_to_text(response))


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
    return _response_to_text(response)
