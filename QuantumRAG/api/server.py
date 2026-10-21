"""FastAPI server: JSON API plus the static dashboard.

Run with:  python run.py   (or  uvicorn api.server:app --port 8000)
"""
from __future__ import annotations

import json
import logging
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from quantumrag.config import RESULTS_DIR, ROOT, QuantumConfig
from quantumrag.engine import QuantumRAG

log = logging.getLogger("quantumrag.api")
WEB_DIR = ROOT / "web"

_engine: QuantumRAG | None = None
_engine_lock = threading.Lock()


def engine() -> QuantumRAG:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = QuantumRAG()
        return _engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=engine, daemon=True).start()   # warm up in the background
    yield


app = FastAPI(title="QuantumRAG", version="1.0.0", lifespan=lifespan)


# ----------------------------------------------------------------- models
class QuantumParams(BaseModel):
    n_qubits: int = Field(8, ge=2, le=12)
    encoding: Literal["angle", "iqp", "amplitude"] = "amplitude"
    method: Literal["kernel", "swap", "iqae"] = "kernel"
    layers: int = Field(2, ge=1, le=6)
    bandwidth: float = Field(1.0, ge=0.05, le=4.0)
    shots: Optional[int] = Field(None, ge=1, le=1_000_000)
    execution: Literal["statevector", "circuit"] = "statevector"
    fusion: float = Field(0.0, ge=0.0, le=1.0)
    iqae_epsilon: float = Field(0.01, gt=0.0005, lt=0.5)

    def to_config(self) -> QuantumConfig:
        fields = QuantumParams.model_fields.keys()
        return QuantumConfig(**self.model_dump(include=set(fields))).validate()


class SearchRequest(QuantumParams):
    query: str = Field(..., min_length=1, max_length=2000)
    pipelines: list[Literal["classical", "quantum", "hybrid"]] = ["classical", "quantum", "hybrid"]
    k: int = Field(5, ge=1, le=50)
    candidates: int = Field(50, ge=2, le=500)


class AnswerRequest(SearchRequest):
    pipeline: Literal["classical", "quantum", "hybrid"] = "hybrid"
    provider: Optional[Literal["extractive", "ollama", "openai"]] = None


class CircuitRequest(QuantumParams):
    query: Optional[str] = None
    doc_index: Optional[int] = Field(None, ge=0)


class NewDocument(BaseModel):
    title: str = Field("", max_length=300)
    text: str = Field(..., min_length=20, max_length=200_000)


class CorpusAdd(BaseModel):
    documents: list[NewDocument] = Field(..., min_length=1, max_length=200)


class UploadedFile(BaseModel):
    name: str = Field(..., min_length=1, max_length=300)
    data_base64: str = Field(..., min_length=4)


class CorpusUpload(BaseModel):
    files: list[UploadedFile] = Field(..., min_length=1, max_length=50)
    mode: Literal["add", "replace"] = "add"


class BenchmarkRequest(BaseModel):
    n_queries: int = Field(200, ge=10, le=3000)
    candidates: int = Field(50, ge=5, le=500)
    analysis: bool = True


# ----------------------------------------------------------------- errors
@app.exception_handler(ValueError)
async def value_error_handler(_req, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _guard_circuit(req: QuantumParams, docs: int, pipelines) -> None:
    if req.execution != "circuit":
        return
    if "quantum" in pipelines and docs > 150:
        raise HTTPException(400, "Circuit execution runs one simulated circuit per document. "
                                 "Use it with the hybrid pipeline only, or use statevector "
                                 "execution for the full-corpus quantum pipeline.")
    if req.method == "swap" and req.n_qubits > 6:
        raise HTTPException(400, "Circuit-mode SWAP test needs 2N+1 qubits; use at most 6 qubits.")


# --------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"ok": True, "ready": _engine is not None}


@app.get("/api/status")
def status():
    return engine().status()


@app.get("/api/samples")
def samples(n: int = 6, seed: int = 7):
    e = engine()
    if not e.queries:
        return {"samples": []}
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(e.queries), size=min(n, len(e.queries)), replace=False)
    by_id = {d.id: d for d in e.documents}
    out = []
    for i in pick:
        q = e.queries[int(i)]
        doc = by_id.get(q.relevant[0])
        out.append({"query": q.text, "answer": q.answer, "relevant": q.relevant,
                    "relevant_title": doc.title if doc else None})
    return {"samples": out}


@app.post("/api/search")
def search(req: SearchRequest):
    e = engine()
    _guard_circuit(req, len(e.documents), req.pipelines)
    res = e.search(req.query, req.pipelines, req.k, req.to_config(), req.candidates)
    gold = [q for q in e.queries if q.text.strip().lower() == req.query.strip().lower()]
    if gold:
        res["relevant"] = gold[0].relevant
    return res


@app.post("/api/answer")
def answer(req: AnswerRequest):
    e = engine()
    _guard_circuit(req, len(e.documents), [req.pipeline])
    return e.answer(req.query, req.pipeline, req.k, req.to_config(), req.candidates, req.provider)


@app.post("/api/circuit")
def circuit(req: CircuitRequest):
    e = engine()
    if req.doc_index is not None and req.doc_index >= len(e.documents):
        raise HTTPException(404, "document index out of range")
    return e.circuit(req.to_config(), req.query, req.doc_index)


@app.get("/api/corpus")
def corpus(offset: int = 0, limit: int = 25, q: str = ""):
    e = engine()
    docs = list(enumerate(e.documents))
    if q:
        ql = q.lower()
        docs = [(i, d) for i, d in docs if ql in d.title.lower() or ql in d.text.lower()]
    page = docs[offset: offset + limit]
    topics: dict[str, int] = {}
    for d in e.documents:
        topics[d.topic or "Other"] = topics.get(d.topic or "Other", 0) + 1
    return {"total": len(docs), "offset": offset, "limit": limit,
            "items": [{"index": i, **d.to_dict(snippet=280), "words": len(d.text.split())} for i, d in page],
            "topics": sorted(({"topic": t, "count": c} for t, c in topics.items()),
                             key=lambda r: -r["count"])}


@app.get("/api/corpus/{index}")
def corpus_doc(index: int):
    e = engine()
    if not 0 <= index < len(e.documents):
        raise HTTPException(404, "document not found")
    d = e.documents[index]
    return {"index": index, **d.to_dict()}


@app.post("/api/corpus")
def corpus_add(body: CorpusAdd):
    added = engine().add_documents([d.model_dump() for d in body.documents])
    return {"added_chunks": added, "documents": len(engine().documents)}


@app.post("/api/corpus/upload")
def corpus_upload(body: CorpusUpload):
    """Extract text from .txt / .md / .pdf / .docx files and index it."""
    import base64
    import binascii
    from quantumrag.ingest import ExtractionError, extract

    items, report, errors = [], [], []
    for f in body.files:
        try:
            data = base64.b64decode(f.data_base64, validate=True)
        except (binascii.Error, ValueError):
            errors.append(f"{f.name}: the upload was corrupted, try again")
            continue
        try:
            sections = extract(f.name, data)
        except ExtractionError as exc:
            errors.append(str(exc))
            continue
        words = sum(len(sec.text.split()) for sec in sections)
        report.append({"name": f.name, "sections": len(sections), "words": words})
        items += [{"title": sec.title, "text": sec.text, "topic": f.name} for sec in sections]
    if not items:
        raise HTTPException(400, " ".join(errors) or "No text could be extracted from the files.")
    if body.mode == "replace" and sum(len(i["text"].split()) for i in items) < 40:
        raise HTTPException(400, "There is too little text to search on its own; add these files to the corpus instead.")
    added = engine().add_documents(items, replace=body.mode == "replace")
    return {"files": report, "errors": errors, "added_chunks": added,
            "documents": len(engine().documents), "corpus": engine().corpus_kind}


@app.post("/api/corpus/reset")
def corpus_reset():
    engine().reset()
    return {"documents": len(engine().documents)}


# --------------------------------------------------------------- benchmark
_bench = {"running": False, "step": 0, "total": 0, "label": "", "error": None}


def _run_benchmark(n_queries: int, candidates: int, analysis: bool):
    from quantumrag.evaluation import run_benchmark

    def progress(i, total, label):
        _bench.update(step=i, total=total, label=label)
    try:
        run_benchmark(engine(), n_queries=n_queries, candidates=candidates,
                      include_analysis=analysis, progress=progress)
    except Exception as exc:  # pragma: no cover
        _bench["error"] = f"{exc.__class__.__name__}: {exc}"
        log.error(traceback.format_exc())
    finally:
        _bench["running"] = False


@app.get("/api/benchmark")
def benchmark_latest():
    path = RESULTS_DIR / "benchmark_latest.json"
    if not path.exists():
        return {"available": False}
    return {"available": True, **json.loads(path.read_text())}


@app.post("/api/benchmark/run")
def benchmark_run(req: BenchmarkRequest):
    if _bench["running"]:
        raise HTTPException(409, "a benchmark is already running")
    if not engine().queries:
        raise HTTPException(400, "Benchmarks need the labelled SQuAD questions. Restore the benchmark "
                                 "corpus on the Corpus page, then run the benchmark.")
    _bench.update(running=True, step=0, total=1, label="starting", error=None)
    threading.Thread(target=_run_benchmark, args=(req.n_queries, req.candidates, req.analysis),
                     daemon=True).start()
    return {"started": True}


@app.get("/api/benchmark/progress")
def benchmark_progress():
    return _bench


@app.get("/api/crosscheck")
def crosscheck(encoding: str = "angle", n_qubits: int = 4, pairs: int = 5):
    from quantumrag.quantum import qiskit_backend as qb
    if not qb.HAVE_QISKIT:
        return {"available": False, "detail": "Qiskit is not installed. Run: pip install qiskit"}
    if encoding not in ("angle", "iqp", "amplitude") or not 2 <= n_qubits <= 6:
        raise HTTPException(400, "encoding must be angle/iqp/amplitude and 2 <= n_qubits <= 6")
    cfg = QuantumConfig(n_qubits=n_qubits, encoding=encoding).validate()
    sp = engine().space(cfg)
    return {"available": True, **qb.cross_check(sp.doc_features, n_qubits, encoding, cfg.layers,
                                                cfg.bandwidth, pairs)}


# ------------------------------------------------------------------ static
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")
