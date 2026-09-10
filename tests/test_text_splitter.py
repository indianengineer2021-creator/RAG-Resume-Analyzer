from langchain_core.documents import Document

from src.text_splitter import chunk_documents


def test_chunk_documents_returns_multiple_chunks():
    docs = [
        Document(
            page_content=(
                "Python developer with 5 years of experience. "
                "Worked on data pipelines, machine learning, cloud services. "
                "Built dashboards and automated deployments. "
                "Strong SQL and ETL experience. "
                "Managed stakeholder communication and production systems. "
                "Led analytics implementation using cloud infrastructure. "
                "Built AI applications and collaborated with product teams. "
            ),
            metadata={"source": "example.pdf", "document_type": "resume"},
        )
    ]

    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(chunk.metadata.get("chunk_id") for chunk in chunks)
    assert chunks[0].metadata["source"] == "example.pdf"
