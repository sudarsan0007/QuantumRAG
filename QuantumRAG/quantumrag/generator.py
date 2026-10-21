"""Context building and answer generation.

Providers
---------
extractive (default)  Offline. Selects the retrieved sentences most similar to the
                      question using the shared embedder and returns them with
                      source citations. No data leaves the machine.
ollama                Local LLM through an Ollama server (e.g. llama3, mistral).
                      Set QRAG_OLLAMA_URL and QRAG_OLLAMA_MODEL to override.
openai                Any OpenAI-compatible chat endpoint. Set QRAG_OPENAI_BASE_URL,
                      QRAG_OPENAI_API_KEY and QRAG_OPENAI_MODEL. Sends the retrieved
                      context to that endpoint, so it is opt-in only.

If a remote provider fails, the extractive answer is returned with a note.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np

from .data import Document

PROMPT = ("You are a careful assistant. Answer the question using only the numbered "
          "context passages. Cite passages like [1]. If the answer is not in the context, "
          "say so.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:")


class ContextBuilder:
    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars

    def build(self, docs: list[Document]) -> str:
        parts, used = [], 0
        for i, d in enumerate(docs, 1):
            block = f"[{i}] {d.title}\n{d.text}"
            if used + len(block) > self.max_chars and parts:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)


_STOP = set("""a an the of in on at to for from by with and or but is are was were be been being
do does did what which who whom whose when where why how that this these those it its as
into than then there their they them he she his her we you your i not no can could would
should will shall may might has have had about after before during over under also such
many much more most some any each other one two name type kind""".split())


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOP}


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


_ABBREV = re.compile(r"(?:\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Mt|Fig|Figs|Eq|No|Vol|vs|etc|al|Inc|Ltd|Co|Corp|Dept|Univ|approx|e\.g|i\.e)|\b[A-Z])\.$")


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends, but not after titles, initials or common abbreviations."""
    parts = [p.strip() for p in _SENT.split(text) if p.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and _ABBREV.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [m for m in merged if len(m) > 20]


class LLMGenerator:
    def __init__(self, embedder, provider: str | None = None):
        self.embedder = embedder
        self.provider = provider or os.environ.get("QRAG_LLM", "extractive")
        self.context_builder = ContextBuilder()

    @property
    def model_name(self) -> str:
        return self.model_for(self.provider)

    @staticmethod
    def model_for(provider: str) -> str:
        """Model name for a given provider (the provider may differ per request)."""
        if provider == "ollama":
            return os.environ.get("QRAG_OLLAMA_MODEL", "llama3.2")
        if provider == "openai":
            return os.environ.get("QRAG_OPENAI_MODEL", "gpt-4o-mini")
        return "extractive (embedding sentence selection)"

    def generate(self, question: str, docs: list[Document], scores: list[float] | None = None,
                 provider: str | None = None) -> dict:
        provider = provider or self.provider
        t0 = time.perf_counter()
        note = None
        if provider in ("ollama", "openai"):
            try:
                context = self.context_builder.build(docs)
                text = self._ollama(question, context) if provider == "ollama" else self._openai(question, context)
                return {"answer": text.strip(), "provider": provider, "model": self.model_for(provider),
                        "confidence": float(np.mean(scores)) if scores else None,
                        "evidence": [], "latency_ms": (time.perf_counter() - t0) * 1e3, "note": None}
            except Exception as exc:  # network or server not running
                if provider == "ollama":
                    model = self.model_for("ollama")
                    if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                        note = (f"Ollama is running, but the model '{model}' is not installed. "
                                f"Run: ollama pull {model} (or set QRAG_OLLAMA_MODEL to a model you have). "
                                "Showing the offline extractive answer instead.")
                    else:
                        note = (f"Ollama is not reachable ({exc.__class__.__name__}). Make sure the Ollama app "
                                "is running. Showing the offline extractive answer instead.")
                else:
                    note = (f"The OpenAI-compatible API call failed ({exc.__class__.__name__}). Check "
                            "QRAG_OPENAI_API_KEY. Showing the offline extractive answer instead.")
        out = self._extractive(question, docs)
        out.update(latency_ms=(time.perf_counter() - t0) * 1e3, note=note)
        return out

    # ------------------------------------------------------------ providers
    def _extractive(self, question: str, docs: list[Document], max_sent: int = 3) -> dict:
        """Answer with the best-supported sentences, all taken from ONE passage.

        Each sentence is scored by embedding similarity to the question plus the share of
        the question's content words it contains. The best sentence decides the source
        passage; up to two more sentences from that same passage are added only if they
        score close to it. Sentences are never stitched across unrelated passages.
        """
        cands = []
        for di, d in enumerate(docs[:5]):
            for si, sent in enumerate(split_sentences(d.text) or [d.text]):
                cands.append((di, si, sent))
        if not cands:
            return {"answer": "No retrieved context to answer from.", "provider": "extractive",
                    "model": self.model_for("extractive"), "confidence": 0.0, "evidence": []}
        q_terms = _content_words(question)
        q = self.embedder.embed(question)
        sims = self.embedder.embed([c[2] for c in cands]) @ q
        overlap = np.array([len(q_terms & _content_words(c[2])) / max(len(q_terms), 1) for c in cands])
        rank_prior = np.array([0.04 * (4 - min(c[0], 4)) / 4 for c in cands])
        score = sims + 0.35 * overlap + rank_prior

        best = int(np.argmax(score))
        src = cands[best][0]
        same = [i for i, c in enumerate(cands) if c[0] == src and i != best
                and score[i] >= 0.8 * score[best]]
        same = sorted(same, key=lambda i: -score[i])[:max_sent - 1]
        chosen = sorted([best] + same, key=lambda i: cands[i][1])
        answer = " ".join(cands[i][2] for i in chosen) + f" [{src + 1}]"
        evidence = [{"source": src + 1, "sentence": cands[i][2], "score": float(score[i])} for i in chosen]
        return {"answer": answer, "provider": "extractive", "model": self.model_for("extractive"),
                "confidence": float(np.clip(sims[best], 0, 1)), "evidence": evidence}

    def _ollama(self, question: str, context: str) -> str:
        url = os.environ.get("QRAG_OLLAMA_URL", "http://localhost:11434") + "/api/generate"
        body = json.dumps({"model": self.model_for("ollama"), "stream": False,
                           "prompt": PROMPT.format(context=context, question=question)}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["response"]

    def _openai(self, question: str, context: str) -> str:
        base = os.environ.get("QRAG_OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ["QRAG_OPENAI_API_KEY"]
        body = json.dumps({"model": self.model_for("openai"), "temperature": 0,
                           "messages": [{"role": "user", "content": PROMPT.format(context=context, question=question)}]
                           }).encode()
        req = urllib.request.Request(base + "/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
