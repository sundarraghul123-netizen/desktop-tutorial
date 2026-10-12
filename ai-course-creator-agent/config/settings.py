import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

def _p(key, default):
    p = Path(os.getenv(key, default))
    return p if p.is_absolute() else ROOT / p

API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
DB_PATH = _p("DATABASE_PATH", "./storage/course_creator.db")
CHROMA_PATH = _p("CHROMA_PATH", "./storage/chroma")
EXPORT_DIR = _p("EXPORT_DIR", "./storage/exports")
TOP_K = int(os.getenv("TOP_K", 5))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", 0.35))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", 60000))
MODEL_CONCURRENCY = max(1, int(os.getenv("GEMINI_MAX_CONCURRENCY", 6)))
MODEL_REQUEST_INTERVAL = max(0.1, float(os.getenv("GEMINI_MIN_REQUEST_INTERVAL", 4.1)))
MODEL_REQUEST_TIMEOUT_MS = max(1000, int(os.getenv("GEMINI_REQUEST_TIMEOUT_MS", 120000)))
for d in (DB_PATH.parent, CHROMA_PATH, EXPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)
