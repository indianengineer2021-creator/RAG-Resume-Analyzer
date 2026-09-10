import logging
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_DB_PATH, RESUME_COLLECTION_NAME
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def create_vector_store(collection_name: str = RESUME_COLLECTION_NAME, persist_directory: Optional[str | Path] = None):
    """Create or connect to a persistent ChromaDB vector store."""
    path = Path(persist_directory) if persist_directory else CHROMA_DB_PATH
    path.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to ChromaDB at %s", path)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(path),
    )


def add_documents(documents: Iterable[Document], collection_name: str = RESUME_COLLECTION_NAME):
    """Add documents to the ChromaDB collection and persist them to disk."""
    vector_store = create_vector_store(collection_name=collection_name)
    vector_store.add_documents(list(documents))
    vector_store.persist()
    logger.info("Added %s documents to ChromaDB collection %s", len(list(documents)), collection_name)
    return vector_store


def load_vector_store(collection_name: str = RESUME_COLLECTION_NAME, persist_directory: Optional[str | Path] = None):
    """Load an existing ChromaDB store from disk."""
    path = Path(persist_directory) if persist_directory else CHROMA_DB_PATH
    logger.info("Loading ChromaDB from %s", path)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(path),
    )


def clear_vector_store(collection_name: str = RESUME_COLLECTION_NAME, persist_directory: Optional[str | Path] = None):
    """Delete the persisted Chroma collection for this app."""
    path = Path(persist_directory) if persist_directory else CHROMA_DB_PATH
    store = load_vector_store(collection_name=collection_name, persist_directory=path)
    store.delete_collection()
    logger.info("Cleared ChromaDB collection %s", collection_name)
    return True
