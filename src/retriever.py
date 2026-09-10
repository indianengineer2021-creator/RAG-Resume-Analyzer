import logging
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.config import DEFAULT_RETRIEVAL_K

logger = logging.getLogger(__name__)


def build_retriever(vector_store: VectorStore, k: int = DEFAULT_RETRIEVAL_K):
    """Create a semantic similarity retriever for the stored resume content."""
    # Retrieval is the key step in a RAG system: instead of sending the whole resume to the LLM,
    # we fetch only the most relevant chunks based on semantic similarity to the JD or question.
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_relevant_documents(vector_store: VectorStore, query: str, k: int = DEFAULT_RETRIEVAL_K) -> List[Document]:
    """Return the most relevant chunks for a given question or job description."""
    retriever = build_retriever(vector_store, k=k)
    logger.info("Retrieving top %s document chunks for query: %s", k, query[:80])
    return retriever.invoke(query)
