from typing import List, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def chunk_documents(
    documents: Sequence[Document], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """Split resume documents into chunks while retaining source metadata and chunk numbering."""
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False,
    )

    chunks = text_splitter.split_documents(list(documents))
    for index, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.metadata or {})
        metadata["chunk_id"] = index
        chunk.metadata = metadata

    return chunks
