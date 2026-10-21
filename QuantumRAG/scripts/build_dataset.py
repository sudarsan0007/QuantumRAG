"""Build the benchmark corpus from the official SQuAD v1.1 dev set.

Each SQuAD paragraph ("context") becomes one retrievable document. Each question
is a query whose single relevant document is the paragraph it was written for.
This gives exact relevance labels for Recall@K, NDCG@K and MRR.

Usage:
    python scripts/build_dataset.py --squad path/to/dev-v1.1.json --docs 600

The SQuAD dev file can be downloaded from
https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(squad_path: Path, n_docs: int, seed: int) -> dict:
    data = json.loads(squad_path.read_text(encoding="utf-8"))["data"]
    rng = random.Random(seed)

    paragraphs = []
    for article in data:
        title = article["title"].replace("_", " ")
        for p_idx, para in enumerate(article["paragraphs"]):
            paragraphs.append((title, p_idx, para))

    # Stratified sample: spread documents evenly across articles so the corpus
    # covers many topics instead of a few long articles.
    by_title: dict[str, list] = {}
    for item in paragraphs:
        by_title.setdefault(item[0], []).append(item)
    titles = sorted(by_title)
    for t in titles:
        rng.shuffle(by_title[t])

    chosen = []
    while len(chosen) < n_docs and any(by_title[t] for t in titles):
        for t in titles:
            if by_title[t] and len(chosen) < n_docs:
                chosen.append(by_title[t].pop())

    documents, queries = [], []
    for title, p_idx, para in chosen:
        doc_id = f"{title.lower().replace(' ', '-')}-{p_idx:03d}"
        documents.append({"id": doc_id, "title": f"{title}, paragraph {p_idx + 1}",
                          "topic": title, "text": para["context"].strip()})
        for qa in para["qas"]:
            queries.append({"id": qa["id"], "text": qa["question"].strip(),
                            "relevant": [doc_id],
                            "answer": qa["answers"][0]["text"] if qa["answers"] else ""})

    return {
        "name": "SQuAD v1.1 dev (paragraph retrieval subset)",
        "source": "https://rajpurkar.github.io/SQuAD-explorer/",
        "license": "CC BY-SA 4.0",
        "seed": seed,
        "documents": documents,
        "queries": queries,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", type=Path, required=True)
    ap.add_argument("--docs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "squad_subset.json")
    args = ap.parse_args()
    ds = build(args.squad, args.docs, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ds, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(ds['documents'])} documents and {len(ds['queries'])} queries to {args.out}")


if __name__ == "__main__":
    main()
