"""
embed_cache.py
Handles all embedding computation and vector search for the AI Twin.
- Uses Groq API when GROQ_API_KEY + GROQ_EMBED_MODEL are set
- Falls back to local SentenceTransformer (all-MiniLM-L6-v2) otherwise
- Saves / loads cache from  <project>/data/embeddings.npz + metadata.json
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    from groq import Groq, APIError, RateLimitError, APIConnectionError
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# ── logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── paths  (data/ lives NEXT TO this file) ────────────────────────────────────
DATA_DIR  = Path(__file__).resolve().parent / "data"
EMB_FILE  = DATA_DIR / "embeddings.npz"
META_FILE = DATA_DIR / "metadata.json"

# ── constants ──────────────────────────────────────────────────────────────────
_DEFAULT_EMBED_MODEL = "nomic-embed-text-v1.5"
_LOCAL_MODEL_NAME    = "all-MiniLM-L6-v2"
MAX_TEXT_LENGTH      = 8_192
MAX_BATCH_SIZE       = 96
MAX_QUERY_LENGTH     = 1_024
RETRY_ATTEMPTS       = 3
RETRY_BASE_DELAY     = 1.0      # seconds, doubles each retry

# ── module-level cache ─────────────────────────────────────────────────────────
_groq_client:  Optional[Any] = None
_local_model:  Optional[Any] = None


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _groq_embed_model() -> str:
    """Return the embedding model name, preferring the env var."""
    return os.getenv("GROQ_EMBED_MODEL", _DEFAULT_EMBED_MODEL).strip()


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_AVAILABLE:
            raise RuntimeError("groq package not installed. Run: pip install groq")
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            raise ValueError("GROQ_API_KEY is not set.")
        if not key.startswith("gsk_"):
            raise ValueError("GROQ_API_KEY looks malformed (expected gsk_...).")
        _groq_client = Groq(api_key=key)
    return _groq_client


def _get_local_model():
    global _local_model
    if _local_model is None:
        if not ST_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        logger.info("Loading local embedding model '%s'…", _LOCAL_MODEL_NAME)
        _local_model = SentenceTransformer(_LOCAL_MODEL_NAME)
    return _local_model


def _sanitise(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.split())[:max_len]


def _safe_path(path: Path) -> Path:
    resolved = path.resolve()
    if not str(resolved).startswith(str(DATA_DIR.resolve())):
        raise ValueError(f"Path '{path}' escapes DATA_DIR — blocked.")
    return resolved


# ══════════════════════════════════════════════════════════════════════════════
# Embedding backends
# ══════════════════════════════════════════════════════════════════════════════

def _embed_groq(texts: List[str]) -> np.ndarray:
    client = _get_groq_client()
    model  = _groq_embed_model()
    all_vecs: List[List[float]] = []

    for start in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[start : start + MAX_BATCH_SIZE]

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                resp    = client.embeddings.create(model=model, input=batch)
                ordered = sorted(resp.data, key=lambda x: x.index)
                all_vecs.extend(item.embedding for item in ordered)
                break
            except RateLimitError as exc:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning("Rate-limit (attempt %d/%d) – retry in %.1fs: %s",
                               attempt, RETRY_ATTEMPTS, delay, exc)
                if attempt == RETRY_ATTEMPTS:
                    raise
                time.sleep(delay)
            except APIConnectionError as exc:
                logger.error("Groq connection error: %s", exc)
                raise
            except APIError as exc:
                logger.error("Groq API error %s: %s", exc.status_code, exc)
                raise

    return np.array(all_vecs, dtype=np.float32)


def _embed_local(texts: List[str]) -> np.ndarray:
    model = _get_local_model()
    return model.encode(texts, convert_to_numpy=True,
                        show_progress_bar=False).astype(np.float32)


def compute_embeddings(texts: List[str]) -> np.ndarray:
    """Embed texts via Groq if key is available, else local SentenceTransformer."""
    if not texts:
        return np.empty((0,), dtype=np.float32)

    clean = [_sanitise(t) for t in texts]

    if os.getenv("GROQ_API_KEY", "").strip() and GROQ_AVAILABLE:
        logger.info("Embedding %d texts via Groq (%s)…", len(clean), _groq_embed_model())
        return _embed_groq(clean)

    logger.info("Embedding %d texts via local model…", len(clean))
    return _embed_local(clean)


# ══════════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_embeddings(
    vectors:  np.ndarray,
    metadata: List[Dict[str, Any]],
    emb_path:  Path = EMB_FILE,
    meta_path: Path = META_FILE,
) -> None:
    """
    Write embeddings (key='arr') and metadata to disk.
    Always creates DATA_DIR first.
    """
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {vectors.shape}.")
    if len(vectors) != len(metadata):
        raise ValueError(
            f"Vector count {len(vectors)} != metadata count {len(metadata)}."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    emb_path  = _safe_path(emb_path)
    meta_path = _safe_path(meta_path)

    try:
        np.savez_compressed(emb_path, arr=vectors)
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)

        if not emb_path.exists() or not meta_path.exists():
            raise OSError("Files not found after write — disk may be full.")

        logger.info(
            "✅ Saved %d embeddings → %s (%.1f KB)",
            len(vectors), emb_path, emb_path.stat().st_size / 1024,
        )
    except OSError as exc:
        logger.error("Write failed: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# Item validation
# ══════════════════════════════════════════════════════════════════════════════

def _validate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    seen:  set = set()

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            logger.warning("Index %d: not a dict — skip.", i); continue
        iid = item.get("id")
        if iid is None:
            logger.warning("Index %d: no 'id' — skip.", i); continue
        if iid in seen:
            logger.warning("Duplicate id '%s' at %d — skip.", iid, i); continue
        text = _sanitise(item.get("text", ""))
        if not text:
            logger.warning("Id '%s': empty text — skip.", iid); continue
        seen.add(iid)
        valid.append({
            "id":       iid,
            "text":     text,
            "category": _sanitise(str(item.get("category", "")), max_len=256),
        })

    return valid


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def build_or_load(
    items: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Return (vectors, metadata).
    Loads from cache when valid; otherwise validates, embeds, and saves.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    emb_path  = _safe_path(EMB_FILE)
    meta_path = _safe_path(META_FILE)

    # 1 – try cache
    if emb_path.exists() and meta_path.exists():
        try:
            logger.info("Loading cache from %s …", DATA_DIR)
            with np.load(emb_path) as npz:
                arr = npz["arr"]
            with meta_path.open("r", encoding="utf-8") as fh:
                meta: List[Dict[str, Any]] = json.load(fh)

            if not isinstance(meta, list) or arr.ndim != 2:
                raise ValueError("Cache malformed.")
            if len(arr) != len(meta):
                raise ValueError(f"Count mismatch: {len(arr)} vs {len(meta)}.")

            logger.info("Cache OK — %d items.", len(arr))
            return arr, meta
        except Exception as exc:
            logger.warning("Cache invalid (%s) — rebuilding.", exc)

    # 2 – validate
    clean = _validate_items(items)
    if not clean:
        raise ValueError("No valid items to embed.")

    # 3 – embed
    logger.info("Computing embeddings for %d items…", len(clean))
    vectors = compute_embeddings([it["text"] for it in clean])

    if vectors.shape[0] != len(clean):
        raise RuntimeError(
            f"Embedding mismatch: {vectors.shape[0]} vs {len(clean)}."
        )

    # 4 – save
    save_embeddings(vectors, clean, emb_path, meta_path)
    return vectors, clean


def retrieve(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Return top_k results most similar to query (cosine similarity).
    Raises FileNotFoundError if cache is missing.
    """
    emb_path  = _safe_path(EMB_FILE)
    meta_path = _safe_path(META_FILE)

    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Cache not found in '{DATA_DIR}'. Call build_or_load() first."
        )

    q = _sanitise(query, max_len=MAX_QUERY_LENGTH)
    if not q:
        raise ValueError("Query is empty after sanitisation.")

    with np.load(emb_path) as npz:
        arr: np.ndarray = npz["arr"]
    with meta_path.open("r", encoding="utf-8") as fh:
        meta: List[Dict[str, Any]] = json.load(fh)

    if arr.ndim != 2 or len(arr) == 0:
        raise ValueError("Embedding array empty or wrong shape.")

    qvec = compute_embeddings([q])[0]
    k    = min(max(1, top_k), len(arr))
    nbr  = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nbr.fit(arr)
    dists, idxs = nbr.kneighbors([qvec])

    return [
        {
            "id":       meta[idx].get("id"),
            "text":     meta[idx].get("text", ""),
            "category": meta[idx].get("category", ""),
            "score":    round(float(1.0 - dist), 6),
        }
        for dist, idx in zip(dists[0], idxs[0])
    ]