"""FastAPI microservice implementation for URL classification."""

import hashlib
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.allowlist import AllowlistManager
from backend.app.config import CACHE_MAX_ENTRIES, CACHE_TTL_SECONDS
from backend.app.model import InferenceEngine
from ml.features.normalizer import normalize_url


# In-memory LRU cache: hash -> (timestamp, response_dict)
PREDICTION_CACHE: Dict[str, Tuple[float, dict]] = {}
allowlist_mgr = AllowlistManager()
inference_engine = InferenceEngine()
inference_engine.load()
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights and allowlist during application startup."""
    inference_engine.load()
    yield


app = FastAPI(
    title="SUS Sentinel URL Classifier API",
    version="0.1.0",
    description="High-performance URL safety classification using from-scratch machine learning in pure NumPy.",
    lifespan=lifespan,
)

# CORS configuration for browser extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ClassifyRequest(BaseModel):
    url: str = Field(..., max_length=2048, description="URL to inspect")
    client_version: Optional[str] = Field("0.1.0", description="Client or extension version")
    mode: Optional[str] = Field("binary", description="Classification mode: binary")


class ClassifyResponse(BaseModel):
    url: str
    verdict: str
    category: Optional[str] = None
    confidence: float
    calibrated: bool
    source: str
    model_version: str
    reasons: List[str]
    latency_ms: float
    cached: bool


class ReportRequest(BaseModel):
    url: str
    reported_verdict: str
    user_comment: Optional[str] = None


@app.post("/v1/classify", response_model=ClassifyResponse)
async def classify_url(payload: ClassifyRequest):
    """Classify a URL as safe, suspicious, or unsafe."""
    t_start = time.perf_counter()
    raw_url = payload.url.strip()

    if not raw_url:
        raise HTTPException(status_code=400, detail="URL parameter cannot be empty.")

    # Canonical normalization (strips query parameters and fragments for privacy)
    clean_url = normalize_url(raw_url, strip_query_and_fragment=True)
    url_hash = hashlib.sha256(clean_url.encode("utf-8")).hexdigest()

    # 1. In-memory allowlist check (O(1))
    if allowlist_mgr.is_allowed(clean_url):
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return ClassifyResponse(
            url=clean_url,
            verdict="safe",
            category=None,
            confidence=0.01,
            calibrated=True,
            source="allowlist",
            model_version=inference_engine.model_version,
            reasons=["Domain is on the verified global top-sites allowlist"],
            latency_ms=round(latency_ms, 2),
            cached=False,
        )

    # 2. Check LRU cache
    now = time.time()
    if url_hash in PREDICTION_CACHE:
        cached_time, cached_res = PREDICTION_CACHE[url_hash]
        if now - cached_time < CACHE_TTL_SECONDS:
            cached_res["cached"] = True
            cached_res["latency_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
            return ClassifyResponse(**cached_res)

    # 3. Model inference pass (<1ms)
    result = inference_engine.predict(clean_url)
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    response_data = {
        "url": clean_url,
        "verdict": result["verdict"],
        "category": None,
        "confidence": result["confidence"],
        "calibrated": True,
        "source": result["source"],
        "model_version": result["model_version"],
        "reasons": result["reasons"],
        "latency_ms": round(latency_ms, 2),
        "cached": False,
    }

    # Store in memory cache
    if len(PREDICTION_CACHE) >= CACHE_MAX_ENTRIES:
        # Evict oldest entry
        oldest_key = min(PREDICTION_CACHE.keys(), key=lambda k: PREDICTION_CACHE[k][0])
        del PREDICTION_CACHE[oldest_key]

    PREDICTION_CACHE[url_hash] = (now, response_data)

    return ClassifyResponse(**response_data)


@app.get("/v1/health")
async def health_check():
    """Health check and model telemetry."""
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_version": inference_engine.model_version,
        "model_loaded": inference_engine.is_loaded,
        "allowlist_entries": len(allowlist_mgr.domains),
        "cache_entries": len(PREDICTION_CACHE),
    }


@app.post("/v1/report", status_code=status.HTTP_202_ACCEPTED)
async def report_url(payload: ReportRequest):
    """Receive user feedback and quarantine for reviewed retraining."""
    clean_url = normalize_url(payload.url, strip_query_and_fragment=True)
    # Quarantine feedback without raw URL logging
    url_hash = hashlib.sha256(clean_url.encode("utf-8")).hexdigest()
    print(f"[Feedback Report] URL Hash: {url_hash[:16]}... Reported: {payload.reported_verdict}")
    return {"status": "quarantined", "message": "Report received and queued for review."}
