#!/usr/bin/env python3
"""Chunking benchmark on SQuAD: does a retrieved chunk contain the whole answer?

Needs fastembed and tiktoken. Writes results/chunking.json.

Every strategy is judged twice: at the usual fixed k, and at a fixed token budget, because a fixed
k quietly hands larger chunks more context to work with.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken
from fastembed import TextEmbedding

import chunkers

SQUAD = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
MODEL = "BAAI/bge-small-en-v1.5"
BUDGET = 1000


def load_documents() -> tuple[list[str], list[dict]]:
    """Rebuild each SQuAD article as one document and keep answers as document offsets."""
    cache = Path("data/squad-dev.json")
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(SQUAD, timeout=120) as response:  # noqa: S310
            cache.write_bytes(response.read())
    documents, questions = [], []
    for doc_id, article in enumerate(json.loads(cache.read_text())["data"]):
        text, offset = "", 0
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                spans = [
                    (offset + a["answer_start"], offset + a["answer_start"] + len(a["text"]))
                    for a in qa["answers"]
                ]
                if spans:
                    questions.append({"doc": doc_id, "question": qa["question"], "answers": spans})
            text += paragraph["context"] + "\n\n"
            offset = len(text)
        documents.append(text.rstrip())
    return documents, questions


def contains_answer(span, answers) -> bool:
    return any(span[0] <= a and b <= span[1] for a, b in answers)


def evaluate(name, spans_by_doc, documents, questions, query_vectors, embedder, encoder, ks):
    chunks = [(d, s) for d, spans in enumerate(spans_by_doc) for s in spans]
    texts = [documents[d][s[0] : s[1]] for d, s in chunks]
    started = time.perf_counter()
    vectors = np.array(list(embedder.passage_embed(texts, batch_size=64)))
    embed_seconds = time.perf_counter() - started
    costs = np.array([len(encoder.encode(t, disallowed_special=())) for t in texts])
    scores = query_vectors @ vectors.T
    order = np.argsort(-scores, axis=1)[:, :200]
    hits = {k: 0 for k in ks}
    budget_hits, split = 0, 0
    for row, question in zip(order, questions, strict=True):
        retrievable = any(
            d == question["doc"] and contains_answer(s, question["answers"]) for d, s in chunks
        )
        split += not retrievable
        found_at, used, inside_budget = None, 0, False
        for rank, index in enumerate(row):
            doc, span = chunks[index]
            used += costs[index]
            match = doc == question["doc"] and contains_answer(span, question["answers"])
            if match and found_at is None:
                found_at = rank
                inside_budget = used <= BUDGET
                break
            if used > BUDGET and found_at is None and rank >= max(ks):
                break
        for k in ks:
            hits[k] += found_at is not None and found_at < k
        budget_hits += inside_budget
    n = len(questions)
    return {
        "strategy": name,
        "chunks": len(chunks),
        "mean_tokens": round(float(costs.mean()), 1),
        "answers_split_across_chunks": round(split / n, 4),
        **{f"hit@{k}": round(hits[k] / n, 4) for k in ks},
        f"hit_within_{BUDGET}_tokens": round(budget_hits / n, 4),
        "embed_seconds": round(embed_seconds, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    documents, questions = load_documents()
    questions = random.Random(args.seed).sample(questions, args.questions)
    encoder = tiktoken.get_encoding("cl100k_base")
    embedder = TextEmbedding(MODEL)
    query_vectors = np.array(list(embedder.query_embed([q["question"] for q in questions])))

    def embed(texts):
        return list(embedder.passage_embed(texts, batch_size=64))

    size = [128, 256, 512, 1024]
    strategies = {f"fixed-{s}": (lambda d, s=s: chunkers.fixed(d, encoder, s)) for s in size}
    strategies |= {
        f"fixed-{s}-overlap15%": (lambda d, s=s: chunkers.fixed(d, encoder, s, int(s * 0.15)))
        for s in (256, 512)
    }
    strategies |= {
        f"recursive-{s}": (lambda d, s=s: chunkers.recursive(d, encoder, s))
        for s in (128, 256, 512)
    }
    strategies |= {
        f"sentence-{s}": (lambda d, s=s: chunkers.sentence_packed(d, encoder, s))
        for s in (256, 512)
    }
    strategies |= {
        f"semantic-p{p}": (lambda d, p=p: chunkers.semantic(d, encoder, embed, p)) for p in (90, 95)
    }
    strategies["paragraph (gold boundary)"] = chunkers.paragraphs
    results = []
    for name, fn in strategies.items():
        started = time.perf_counter()
        spans = [fn(d) for d in documents]
        row = evaluate(
            name, spans, documents, questions, query_vectors, embedder, encoder, (1, 5, 10)
        )
        row["chunking_seconds"] = round(time.perf_counter() - started - row["embed_seconds"], 1)
        results.append(row)
        print(row, flush=True)
    Path("results").mkdir(exist_ok=True)
    meta = {
        "model": MODEL,
        "questions": len(questions),
        "documents": len(documents),
        "seed": args.seed,
    }
    Path("results/chunking.json").write_text(
        json.dumps({**meta, "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
