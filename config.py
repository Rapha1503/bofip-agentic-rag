"""
BOFIP RAG Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PDF_DATA_DIR = RAW_DATA_DIR / "pdfs"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", str(DATA_DIR / "chroma_db")))

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, CHROMA_DB_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# BOFIP Data Source
BOFIP_API_BASE = "https://data.economie.gouv.fr/api/v2/catalog/datasets/bofip-impots"
BOFIP_ATTACHMENTS_URL = f"{BOFIP_API_BASE}/attachments/"
BOFIP_DOC_PDF_URL = f"{BOFIP_API_BASE}/attachments/bofip_documentation_pdf"

# LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MAX_CONTEXT_CHUNKS = int(os.getenv("LLM_MAX_CONTEXT_CHUNKS", "14"))
LLM_MAX_CONTEXT_TOKENS = int(os.getenv("LLM_MAX_CONTEXT_TOKENS", "4500"))

# Fallback models list - tries in order until one works
# Each model has different rate limits and token quotas
GROQ_MODELS = [
    {
        "id": "llama-3.3-70b-versatile",
        "name": "Llama 3.3 70B",
        "description": "Best quality, 131K context"
    },
    {
        "id": "llama-3.1-8b-instant",
        "name": "Llama 3.1 8B",
        "description": "Fast and reliable fallback"
    },
    {
        "id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "name": "Llama 4 Scout",
        "description": "Preview model"
    },
]

# Default model (first in list)
GROQ_MODEL = GROQ_MODELS[0]["id"]

# Embedding Configuration
# NOTE: E5 models require prefixes - handled in embeddings.py
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
# Previous: "paraphrase-multilingual-MiniLM-L12-v2" (lighter but worse for domain vocabulary)

# Chunking Configuration
CHUNK_MIN_TOKENS = 200
CHUNK_MAX_TOKENS = 800
CHUNK_TARGET_TOKENS = 500

# Retrieval Configuration
RETRIEVAL_TOP_K = 5  # Reduced from 10 to minimize noise
HYBRID_ALPHA = 0.5  # Balance between dense (1.0) and sparse (0.0)
RERANK_POOL_SIZE = int(os.getenv("RERANK_POOL_SIZE", "30"))  # Cross-encoder candidate pool (tuned)

# Generation faithfulness guardrail
FAITHFULNESS_GUARDRAIL_ENABLED = os.getenv(
    "FAITHFULNESS_GUARDRAIL_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")
FAITHFULNESS_VERIFIER_MODEL = os.getenv("FAITHFULNESS_VERIFIER_MODEL", "llama-3.1-8b-instant")
FAITHFULNESS_MIN_CONFIDENCE = float(os.getenv("FAITHFULNESS_MIN_CONFIDENCE", "0.55"))
FAITHFULNESS_MAX_CONTEXT_CHUNKS = int(os.getenv("FAITHFULNESS_MAX_CONTEXT_CHUNKS", "6"))

# Series keywords for automatic detection
SERIES_KEYWORDS = {
    "RFPI": ["plus-value", "plus value", "immobilier", "immobilière", "cession immeuble", "SCI", "parts de SCI"],
    "TVA": ["tva", "taxe sur la valeur", "taux réduit", "taux normal", "exonération tva"],
    "IR": ["impôt sur le revenu", "revenu imposable", "déclaration de revenus", "quotient familial"],
    "IS": ["impôt sur les sociétés", "bénéfice imposable", "IS"],
    "BIC": ["bénéfices industriels", "micro-bic", "régime réel"],
    "BNC": ["bénéfices non commerciaux", "profession libérale", "micro-bnc"],
    "ENR": ["enregistrement", "droits de mutation", "succession", "donation"],
}

# Cache Configuration
CACHE_TTL_EMBEDDINGS = None  # Never expire
CACHE_TTL_LLM = 86400 * 7  # 7 days
CACHE_TTL_RETRIEVAL = 3600  # 1 hour
