import logging
from typing import Any, Dict

from src.rag_pipeline import answer_question, analyze_resume
from src.vector_store import create_vector_store

logger = logging.getLogger(__name__)


def analyze_resume_against_jd(job_description: str, vector_store=None, k: int = 5) -> Dict[str, Any]:
    """Higher-level business logic for resume-to-job analysis."""
    logger.info("Starting resume analysis against JD.")
    if vector_store is None:
        vector_store = create_vector_store()
    return analyze_resume(job_description=job_description, vector_store=vector_store, k=k)


def answer_resume_question(question: str, vector_store=None, k: int = 5) -> str:
    """Answer a user question using the RAG pipeline and original resume chunks."""
    logger.info("Answering follow-up question using resume context.")
    if vector_store is None:
        vector_store = create_vector_store()
    return answer_question(question=question, vector_store=vector_store, k=k)
