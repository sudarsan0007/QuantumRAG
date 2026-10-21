"""Corpus and query-set loading."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_DATASET


@dataclass
class Document:
    id: str
    title: str
    text: str
    topic: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self, snippet: int | None = None) -> dict:
        text = self.text if snippet is None else _snippet(self.text, snippet)
        return {"id": self.id, "title": self.title, "topic": self.topic, "text": text}


@dataclass
class Query:
    id: str
    text: str
    relevant: list[str]
    answer: str = ""


def _snippet(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + " ..."


def load_dataset(path: Path = DEFAULT_DATASET, max_docs: int | None = None
                 ) -> tuple[list[Document], list[Query], dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = [Document(id=d["id"], title=d.get("title", d["id"]), text=d["text"],
                     topic=d.get("topic", "")) for d in raw["documents"]]
    if max_docs:
        docs = docs[:max_docs]
    keep = {d.id for d in docs}
    queries = [Query(id=q["id"], text=q["text"], relevant=[r for r in q["relevant"] if r in keep],
                     answer=q.get("answer", "")) for q in raw.get("queries", [])]
    queries = [q for q in queries if q.relevant]
    meta = {k: v for k, v in raw.items() if k not in ("documents", "queries")}
    return docs, queries, meta


def chunk_text(text: str, max_words: int = 160, overlap: int = 30) -> list[str]:
    """Split a long user document into overlapping word windows."""
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return [" ".join(words)] if words else []
    out, step = [], max_words - overlap
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
    return out
