"""Classical modules, retrievers, metrics, generator and the HTTP API."""
import numpy as np
import pytest

from quantumrag.config import QuantumConfig
from quantumrag.data import chunk_text
from quantumrag.evaluation import (estimator_efficiency, ndcg_at_k, precision_at_k,
                                   recall_at_k, reciprocal_rank, evaluate_run)
from quantumrag.index import VectorIndex
from quantumrag.reduction import PCAReducer


# ------------------------------------------------------------- metrics
def test_recall_precision():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"c"}, 2) == 0.0
    assert recall_at_k(ranked, {"c"}, 3) == 1.0
    assert recall_at_k(ranked, {"a", "z"}, 4) == 0.5
    assert precision_at_k(ranked, {"a", "b"}, 4) == 0.5


def test_ndcg_known_values():
    assert ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k(["a", "x"], {"a"}, 10) == 1.0
    assert ndcg_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_mrr():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y", "a"], {"a"}, k=2) == 0.0


# ---------------------------------------------------- classical modules
def test_index_returns_exact_top_k(rng):
    v = rng.normal(size=(200, 32)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    idx, scores = VectorIndex(use_faiss=False).build(v).search(v[17], 5)
    assert idx[0] == 17 and np.isclose(scores[0], 1.0, atol=1e-5)
    brute = np.argsort(-(v @ v[17]))[:5]
    assert list(idx) == list(brute)
    assert np.all(np.diff(scores) <= 1e-7)


def test_pca_angle_range_and_dimensions(rng):
    x = rng.normal(size=(100, 64))
    z = PCAReducer(8).fit_transform(x)
    assert z.shape == (100, 8) and z.min() >= 0 and z.max() <= np.pi
    assert 0 < PCAReducer(8).fit(x).explained_variance <= 1


def test_pca_pads_when_corpus_is_small(rng):
    z = PCAReducer(256, angle_range=False).fit_transform(rng.normal(size=(12, 384)))
    assert z.shape == (12, 256) and np.allclose(z[:, 11:], 0.0)
    with pytest.raises(ValueError):
        PCAReducer(4).fit(rng.normal(size=(1, 384)))


def test_chunking_overlaps():
    words = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text(words, max_words=160, overlap=30)
    assert len(chunks) == 3
    assert chunks[1].split()[0] == "w130"


def test_config_validation():
    QuantumConfig(encoding="amplitude", n_qubits=8).validate()
    with pytest.raises(ValueError):
        QuantumConfig(encoding="amplitude", n_qubits=9).validate()
    with pytest.raises(ValueError):
        QuantumConfig(method="grover").validate()
    assert QuantumConfig(encoding="amplitude", n_qubits=5).feature_dim == 32
    assert QuantumConfig(encoding="iqp", n_qubits=5).feature_dim == 5


# ------------------------------------------------------------ retrieval
def test_classical_finds_self(engine):
    d = engine.documents[42]
    emb = engine.embedder.embed(f"{d.title}. {d.text}")
    r = engine.retrieve("classical", emb, 3)
    assert int(r.indices[0]) == 42


@pytest.mark.parametrize("enc,n", [("angle", 4), ("iqp", 4), ("amplitude", 6)])
def test_quantum_pipeline_ranks_by_fidelity(engine, enc, n):
    cfg = QuantumConfig(encoding=enc, n_qubits=n)
    emb = engine.embedder.embed(engine.queries[0].text)
    r = engine.retrieve("quantum", emb, 10, cfg)
    assert len(r.indices) == 10 and np.all(np.diff(r.scores) <= 1e-12)
    assert r.stats["circuits_executed"] == len(engine.documents)


def test_hybrid_reranks_only_candidates(engine):
    cfg = QuantumConfig(encoding="angle", n_qubits=4)
    emb = engine.embedder.embed(engine.queries[3].text)
    shortlist = set(engine.retrieve("classical", emb, 20).indices.tolist())
    r = engine.retrieve("hybrid", emb, 5, cfg, candidates=20)
    assert set(r.indices.tolist()) <= shortlist
    assert r.stats["circuits_executed"] == 20


def test_hybrid_full_fusion_equals_classical_order(engine):
    cfg = QuantumConfig(encoding="angle", n_qubits=4, fusion=1.0)
    emb = engine.embedder.embed(engine.queries[5].text)
    assert list(engine.retrieve("hybrid", emb, 5, cfg, 30).indices) == list(engine.retrieve("classical", emb, 5).indices)


def test_amplitude_ranking_matches_pca_cosine(engine):
    cfg = QuantumConfig(encoding="amplitude", n_qubits=6)
    emb = engine.embedder.embed(engine.queries[8].text)
    q = engine.retrieve("quantum", emb, 3, cfg).indices
    c = engine.retrieve("pca_cosine", emb, 3, cfg).indices
    assert list(q) == list(c)


def test_search_response_shape(engine):
    out = engine.search(engine.queries[0].text, k=4, cfg=QuantumConfig(encoding="angle", n_qubits=4), candidates=10)
    assert set(out["results"]) == {"classical", "quantum", "hybrid"}
    for res in out["results"].values():
        assert len(res["items"]) == 4 and "search" in res["timings_ms"]
    assert 0 < out["quantum_space"]["explained_variance"] < 1


def test_evaluate_run_metrics_are_in_range(engine):
    out = evaluate_run(engine, {"pipeline": "hybrid", "encoding": "angle", "n_qubits": 4}, engine.queries[:20], candidates=10)
    for v in out["metrics"].values():
        assert 0.0 <= v <= 1.0
    assert out["latency_ms"]["p95"] >= out["latency_ms"]["p50"] > 0
    assert out["memory"]["index_bytes"] > 0


def test_classical_baseline_is_strong(engine):
    out = evaluate_run(engine, {"pipeline": "classical"}, engine.queries[:60])
    assert out["metrics"]["recall@10"] > 0.6


def test_estimator_efficiency_shapes():
    est = estimator_efficiency(trials=5)
    assert len(est["sampling"]) > 3 and len(est["iqae"]) > 3
    assert est["iqae"][-1]["mean_abs_error"] < est["iqae"][0]["mean_abs_error"]


# ------------------------------------------------------------ generator
def test_extractive_answer_cites_sources(engine):
    q = engine.queries[0]
    a = engine.answer(q.text, "classical", 3)
    assert a["provider"] == "extractive" and "[" in a["answer"]
    assert len(a["sources"]) == 3


def test_remote_provider_falls_back(engine, monkeypatch):
    monkeypatch.setenv("QRAG_OLLAMA_URL", "http://127.0.0.1:9")
    a = engine.answer(engine.queries[1].text, "classical", 3, provider="ollama")
    assert a["provider"] == "extractive" and a["note"]


# ------------------------------------------------------------------ API
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api import server
    from quantumrag.engine import QuantumRAG
    server._engine = QuantumRAG(max_docs=300)
    with TestClient(server.app) as c:
        yield c


def test_api_status(client):
    s = client.get("/api/status").json()
    assert s["documents"] == 300 and s["device"] == "default.qubit"


def test_api_search_and_errors(client):
    r = client.post("/api/search", json={"query": "Where is the Rhine?", "encoding": "iqp", "n_qubits": 4})
    assert r.status_code == 200 and len(r.json()["results"]["hybrid"]["items"]) == 5
    assert client.post("/api/search", json={"query": "x", "encoding": "amplitude", "n_qubits": 10}).status_code in (400, 422)
    assert client.post("/api/search", json={"query": "x", "execution": "circuit"}).status_code == 400


def test_api_circuit(client):
    c = client.post("/api/circuit", json={"encoding": "angle", "n_qubits": 3, "method": "swap"}).json()
    assert c["n_wires"] == 7 and any(o["gate"] == "CSWAP" for o in c["ops"])
    assert 0 <= c["fidelity"] <= 1


def test_api_corpus_add_and_reset(client):
    before = client.get("/api/corpus").json()["total"]
    r = client.post("/api/corpus", json={"documents": [{"title": "Test note", "text": "Quantum kernels compare states by their overlap. " * 5}]})
    assert r.status_code == 200 and r.json()["documents"] == before + 1
    hit = client.post("/api/search", json={"query": "quantum kernels compare states overlap", "pipelines": ["classical"]}).json()
    assert hit["results"]["classical"]["items"][0]["title"] == "Test note"


# --------------------------------------------------------------- upload
def _docx_bytes(paragraphs):
    import io, zipfile
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = f'<?xml version="1.0"?><w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def _pdf_bytes(pages):
    """Minimal valid PDF with one text line per page (no extra packages needed)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 11 Tf 40 780 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {len(objs)} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = "%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    return out.encode("latin-1")


def test_extract_txt_docx_and_errors():
    from quantumrag.ingest import ExtractionError, extract
    secs = extract("notes.txt", "First line of my notes about qubits.\nIt continues here.".encode())
    assert secs[0].title == "notes" and "continues" in secs[0].text
    secs = extract("thesis.docx", _docx_bytes(["Chapter one introduces quantum retrieval.", "Second paragraph."]))
    assert "quantum retrieval" in secs[0].text
    with pytest.raises(ExtractionError):
        extract("photo.png", b"not text")
    with pytest.raises(ExtractionError):
        extract("broken.docx", b"not a zip")


def test_extract_pdf_pages():
    pytest.importorskip("pypdf")
    from quantumrag.ingest import extract
    secs = extract("paper.pdf", _pdf_bytes(["The SWAP test uses one ancilla qubit to compare states.",
                                            "Amplitude estimation uses Grover iterations to save shots."]))
    assert [s.title for s in secs] == ["paper, page 1", "paper, page 2"]
    assert "ancilla" in secs[0].text


def test_api_upload_replace_then_reset(client):
    import base64
    topics = {
        "kernels.txt": "Quantum kernels compare documents by the fidelity of their encoded quantum states.",
        "swap.txt": "The SWAP test measures an ancilla qubit to estimate the overlap of two states.",
        "cricket.txt": "Cricket is played between two teams of eleven players on an oval field with a pitch.",
        "rice.txt": "Rice paddies are flooded fields where farmers grow rice in South India during monsoon.",
        "trains.txt": "Indian Railways runs express trains that connect Chennai, Bengaluru and Mumbai daily.",
    }
    files = [{"name": n, "data_base64": base64.b64encode((t + " ") .encode() * 3).decode()} for n, t in topics.items()]
    files.append({"name": "rag.docx", "data_base64": base64.b64encode(_docx_bytes(
        ["FAISS is a library for efficient similarity search over dense vectors."] * 3)).decode()})
    r = client.post("/api/corpus/upload", json={"files": files, "mode": "replace"})
    assert r.status_code == 200 and r.json()["corpus"] == "custom" and r.json()["documents"] == 6
    hit = client.post("/api/search", json={"query": "Which library does efficient similarity search over dense vectors?",
                                           "k": 2, "candidates": 4, "n_qubits": 2, "encoding": "amplitude"}).json()
    assert hit["results"]["classical"]["items"][0]["topic"] == "rag.docx"
    assert hit["results"]["hybrid"]["items"][0]["topic"] == "rag.docx"
    assert client.get("/api/samples").json()["samples"] == []
    assert client.post("/api/benchmark/run", json={}).status_code == 400
    assert client.post("/api/corpus/reset").json()["documents"] == 600


def test_sentence_splitter_keeps_titles_together():
    from quantumrag.generator import split_sentences
    out = split_sentences("The guide for the project is Dr. Vanitha M. at VIT. The review happens in the fall semester.")
    assert out[0] == "The guide for the project is Dr. Vanitha M. at VIT."


def test_ollama_request_uses_ollama_model_name(engine, monkeypatch):
    """Selecting Ollama per request must send the Ollama model name, not the default provider's."""
    import json, threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen = {}

    class Fake(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["model"] = body["model"]
            if body["model"] != "llama3.2":
                self.send_response(404); self.end_headers(); return
            out = json.dumps({"response": "Grinko and colleagues proposed it [1]."}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("QRAG_OLLAMA_URL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.delenv("QRAG_OLLAMA_MODEL", raising=False)
    try:
        a = engine.answer(engine.queries[0].text, "classical", 3, provider="ollama")
        assert seen["model"] == "llama3.2"
        assert a["provider"] == "ollama" and a["model"] == "llama3.2" and "[1]" in a["answer"]
        monkeypatch.setenv("QRAG_OLLAMA_MODEL", "not-installed")
        a = engine.answer(engine.queries[0].text, "classical", 3, provider="ollama")
        assert a["provider"] == "extractive" and "ollama pull not-installed" in a["note"]
    finally:
        srv.shutdown()
