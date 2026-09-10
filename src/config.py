import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESUME_DIR = DATA_DIR / "resumes"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Keep the requested default as specified in the project brief while allowing local compatibility overrides.
DEFAULT_RETRIEVAL_K = 5
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

RESUME_COLLECTION_NAME = "resume_documents"

for directory in [RESUME_DIR, CHROMA_DB_PATH]:
    directory.mkdir(parents=True, exist_ok=True)
