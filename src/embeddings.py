import logging
import os
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_embeddings(model_name: Optional[str] = None):
    """Return a reusable Google Gemini embedding model instance."""
    selected_model = model_name or EMBEDDING_MODEL
    logger.info("Initializing embeddings model: %s", selected_model)
    return GoogleGenerativeAIEmbeddings(
        model=selected_model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
