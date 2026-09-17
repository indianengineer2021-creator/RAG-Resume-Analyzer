from typing import List

from langchain_core.documents import Document

from src.rag_pipeline import _safe_json_loads, answer_question
from src.retriever import retrieve_relevant_documents


class FakeRetriever:
    def __init__(self, items: List[Document]):
        self.items = items

    def invoke(self, query: str):
        return self.items


class FakeVectorStore:
    def __init__(self, items: List[Document]):
        self.items = items

    def as_retriever(self, search_kwargs=None):
        return FakeRetriever(self.items)


def test_safe_json_loads_parses_complete_analysis():
    analysis = _safe_json_loads('{"overall_score": 50, "summary": "Good match", "skill_match": []}')

    assert analysis["overall_score"] == 50
    assert analysis["summary"] == "Good match"


def test_retrieve_relevant_documents_returns_content():
    resume_docs = [
        Document(page_content="Python and SQL experience for 5 years", metadata={"source": "example.pdf"}),
        Document(page_content="Project management and stakeholder communication", metadata={"source": "example.pdf"}),
    ]

    vector_store = FakeVectorStore(resume_docs)
    results = retrieve_relevant_documents(vector_store, "Python experience", k=2)

    assert len(results) == 2
    assert "Python" in results[0].page_content


def test_answer_question_uses_resume_context(monkeypatch):
    response = "The resume shows Python and SQL experience."

    class FakeLLM:
        def invoke(self, prompt):
            return type("Response", (), {"content": response})()

    monkeypatch.setattr("src.rag_pipeline.create_vector_store", lambda: FakeVectorStore([
        Document(page_content="Python and SQL experience for 5 years", metadata={"source": "example.pdf"}),
    ]))
    monkeypatch.setattr("src.rag_pipeline.__import__", lambda *args, **kwargs: type("Mod", (), {"get_llm": lambda: FakeLLM()})())

    answer = answer_question("Do I have Python experience?", vector_store=FakeVectorStore([
        Document(page_content="Python and SQL experience for 5 years", metadata={"source": "example.pdf"}),
    ]), k=1)

    assert "Python" in answer
