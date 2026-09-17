import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DEFAULT_RETRIEVAL_K, EMBEDDING_MODEL, GEMINI_MODEL
from src.document_loader import load_jd_document, load_resume_documents
from src.rag_pipeline import analyze_resume, answer_question
from src.text_splitter import chunk_documents
from src.utils import clean_jd_text, clean_text
from src.vector_store import add_documents, clear_vector_store, create_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_uploaded_file(uploaded_file, target_dir: str = "data/resumes") -> Path:
    """Persist an uploaded resume or JD to disk for local processing."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    file_path = target_path / uploaded_file.name
    with file_path.open("wb") as file:
        file.write(uploaded_file.getvalue())
    return file_path


@st.cache_resource
def get_cached_vector_store():
    """Reuse a persistent ChromaDB store across Streamlit reruns."""
    return create_vector_store()


st.set_page_config(page_title="AI Resume Analyzer – RAG Powered", page_icon="📄", layout="wide")

st.title("AI Resume Analyzer – RAG Powered")
st.caption("Retrieval-Augmented Generation pipeline for grounded resume-to-job matching.")

with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("Gemini model", value=GEMINI_MODEL)
    embedding_model = st.text_input("Embedding model", value=EMBEDDING_MODEL)
    retrieval_count = st.slider("Retrieval count", min_value=1, max_value=10, value=DEFAULT_RETRIEVAL_K)

    if st.button("Reset vector database"):
        try:
            clear_vector_store()
            st.success("Vector database reset successfully.")
        except Exception as exc:
            logger.exception("Failed to reset vector database")
            st.error(f"Could not reset vector database: {exc}")

    st.markdown("### Notes")
    st.caption("If your Google API rejects the exact model name, update the model fields in the .env file and rerun the app.")

with st.container():
    resume_file = st.file_uploader("Section 1 – Upload Resume", type=["pdf", "docx"], help="Upload a PDF or DOCX resume.")
    if resume_file is not None:
        st.success("Resume uploaded successfully")

    st.markdown("---")
    jd_text = st.text_area("Section 2 – Paste Job Description", height=220, placeholder="Paste the job description here...")
    jd_upload = st.file_uploader("Optional: Upload JD file", type=["txt", "md", "pdf", "docx"], help="Upload a text, PDF, or DOCX job description.")

    if jd_upload is not None:
        try:
            jd_path = save_uploaded_file(jd_upload, "data/resumes")
            jd_from_file = load_jd_document(jd_path)
            if jd_text.strip():
                jd_text = jd_text + "\n\n" + jd_from_file
            else:
                jd_text = jd_from_file
            st.success("Job description loaded successfully")
        except Exception as exc:
            logger.exception("JD upload failed")
            st.error(f"Could not read the uploaded job description: {exc}")

    st.markdown("---")

    if st.button("Analyze Resume"):
        try:
            if resume_file is None:
                raise ValueError("Please upload a resume before analyzing.")
            if not jd_text or not jd_text.strip():
                raise ValueError("Please provide or upload a job description before analysis.")

            with st.status("Loading resume...", expanded=True) as status:
                resume_path = save_uploaded_file(resume_file)
                docs = load_resume_documents(resume_path)
                resume_document = docs[0]
                cleaned_doc = type(resume_document)(page_content=clean_text(resume_document.page_content), metadata=resume_document.metadata)
                status.update(label="Creating chunks...", state="running")
                chunks = chunk_documents([cleaned_doc], chunk_size=800, chunk_overlap=150)

                status.update(label="Generating embeddings and storing resume context...", state="running")
                clear_vector_store()
                vector_store = add_documents(chunks)

                status.update(label="Searching relevant resume information...", state="running")
                relevant_chunks = vector_store.similarity_search(clean_jd_text(jd_text), k=retrieval_count)
                st.session_state["retrieved_chunks"] = relevant_chunks

                status.update(label="Running Gemini analysis...", state="running")
                results = analyze_resume(jd_text, vector_store=vector_store, k=retrieval_count)
                st.session_state["analysis_result"] = results
                st.session_state["vector_store"] = vector_store
                st.session_state["jd_text"] = jd_text
                status.update(label="Preparing results...", state="complete")

            st.success("Analysis complete.")
        except Exception as exc:
            logger.exception("Resume analysis failed")
            st.error(f"Analysis failed: {exc}")

    if "analysis_result" in st.session_state:
        result = st.session_state["analysis_result"]
        overall_score = int(result.get("overall_score", 0))
        st.markdown("---")
        st.subheader("Overall Match")
        st.metric(label="Match Score", value=f"{overall_score} / 100")
        st.progress(max(0, min(100, overall_score)))

        st.subheader("Executive Summary")
        st.write(result.get("summary", "No summary available."))

        st.subheader("Skills Match")
        skill_rows = result.get("skill_match", [])
        if skill_rows:
            skill_df = pd.DataFrame(skill_rows)
            st.dataframe(skill_df, use_container_width=True)
        else:
            st.write("No skill match details found.")

        missing_skills = result.get("missing_skills", [])
        if missing_skills:
            st.subheader("Missing Skills")
            st.write("\n".join(f"- {skill}" for skill in missing_skills))

        st.subheader("Experience Match")
        st.write(result.get("experience_match", "Not found in resume"))

        st.subheader("Domain Match")
        st.write(result.get("domain_match", "Not found in resume"))

        st.subheader("Certification Match")
        st.write(result.get("certification_match", "Not found in resume"))

        st.subheader("Resume Strengths")
        strengths = result.get("strengths", [])
        if strengths:
            st.write("\n".join(f"- {item}" for item in strengths))
        else:
            st.write("No strengths identified from the retrieved resume context.")

        st.subheader("Resume Weaknesses")
        weaknesses = result.get("weaknesses", [])
        if weaknesses:
            st.write("\n".join(f"- {item}" for item in weaknesses))
        else:
            st.write("No weaknesses identified from the retrieved resume context.")

        st.subheader("ATS Keyword Gaps")
        ats_gaps = result.get("ats_gaps", [])
        if ats_gaps:
            st.write("\n".join(f"- {item}" for item in ats_gaps))
        else:
            st.write("No significant ATS keyword gaps identified.")

        st.subheader("Recommended Resume Improvements")
        recommendations = result.get("recommendations", [])
        if recommendations:
            st.write("\n".join(f"- {item}" for item in recommendations))
        else:
            st.write("No recommendations were generated.")

        st.subheader("Interview Preparation")
        interview_topics = result.get("interview_topics", [])
        if interview_topics:
            st.write("\n".join(f"- {item}" for item in interview_topics))
        else:
            st.write("No interview preparation topics were identified.")

        st.markdown("---")
        with st.expander("Retrieved Resume Context", expanded=False):
            chunks = st.session_state.get("retrieved_chunks", [])
            if chunks:
                st.write(f"Number of chunks retrieved: {len(chunks)}")
                for index, chunk in enumerate(chunks, start=1):
                    st.markdown(f"### Chunk {index}")
                    st.caption(f"Source: {chunk.metadata.get('source', 'resume')}")
                    st.write(chunk.page_content)
            else:
                st.write("No retrieved chunks available.")

        st.markdown("---")
        st.subheader("Ask a question about my resume")
        follow_up_question = st.text_input("Question", placeholder="Do I have enough Python experience for this role?")
        if st.button("Ask") and follow_up_question.strip():
            try:
                vector_store = st.session_state.get("vector_store")
                answer = answer_question(follow_up_question, vector_store=vector_store, k=retrieval_count)
                st.write(answer)
            except Exception as exc:
                logger.exception("Follow-up question failed")
                st.error(f"Could not answer the question: {exc}")
