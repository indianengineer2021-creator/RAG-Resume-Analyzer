import logging
import os
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def _format_model_name(model_name: str) -> str:
    """Use the resource-name format expected by the Google embedding API."""
    return model_name if model_name.startswith("models/") else f"models/{model_name}"


def get_embeddings(model_name: Optional[str] = None):
    """Return a reusable Google Gemini embedding model instance."""
    selected_model = _format_model_name(model_name or EMBEDDING_MODEL)
    logger.info("Initializing embeddings model: %s", selected_model)
    return GoogleGenerativeAIEmbeddings(
        model=selected_model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
