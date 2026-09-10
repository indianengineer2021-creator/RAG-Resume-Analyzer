import logging
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_MODEL

logger = logging.getLogger(__name__)


def get_llm(model_name: str | None = None, temperature: float = 0.2, max_output_tokens: int = 1024):
    """Initialize the Gemini model for grounded resume evaluation."""
    selected_model = model_name or os.getenv("GEMINI_MODEL", GEMINI_MODEL)
    logger.info("Initializing Gemini model: %s", selected_model)
    return ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
