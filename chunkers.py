"""Chunking strategies. Each returns (start, end) character offsets into the document."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

Span = tuple[int, int]
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _token_offsets(text: str, encoder) -> list[int]:
    """The character offset where each token starts."""
    ids = encoder.encode(text, disallowed_special=())
    offsets, position = [], 0
    for piece in encoder.decode_tokens_bytes(ids):
        offsets.append(position)
        position += len(piece.decode("utf-8", errors="ignore"))
    return offsets


def fixed(text: str, encoder, size: int, overlap: int = 0) -> list[Span]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    offsets = _token_offsets(text, encoder)
    spans = []
    for start in range(0, len(offsets), size - overlap):
        end = start + size
        spans.append((offsets[start], offsets[end] if end < len(offsets) else len(text)))
        if end >= len(offsets):
            break
    return spans


def _sentences(text: str, start: int, end: int) -> list[Span]:
    spans, position = [], start
    for match in SENTENCE.finditer(text, start, end):
        spans.append((position, match.start()))
        position = match.end()
    if position < end:
        spans.append((position, end))
    return spans


def _paragraphs(text: str) -> list[Span]:
    spans, position = [], 0
    for match in re.finditer(r"\n\n+", text):
        spans.append((position, match.start()))
        position = match.end()
    if position < len(text):
        spans.append((position, len(text)))
    return spans


def paragraphs(text: str, encoder=None) -> list[Span]:
    return _paragraphs(text)


def _greedy_pack(text: str, units: Sequence[Span], encoder, size: int) -> list[Span]:
    """Join consecutive units until the next one would pass `size` tokens."""
    spans: list[Span] = []
    current: Span | None = None
    used = 0
    for unit in units:
        cost = len(encoder.encode(text[unit[0] : unit[1]], disallowed_special=()))
        if current is not None and used + cost > size:
            spans.append(current)
            current, used = None, 0
        current = (current[0], unit[1]) if current else unit
        used += cost
    if current:
        spans.append(current)
    return spans


def recursive(text: str, encoder, size: int) -> list[Span]:
    """Paragraphs first, sentences inside oversized paragraphs, hard cuts as a last resort."""
    units: list[Span] = []
    for start, end in _paragraphs(text):
        if len(encoder.encode(text[start:end], disallowed_special=())) <= size:
            units.append((start, end))
            continue
        for s_start, s_end in _sentences(text, start, end):
            if len(encoder.encode(text[s_start:s_end], disallowed_special=())) <= size:
                units.append((s_start, s_end))
            else:
                units += [
                    (s_start + a, s_start + b) for a, b in fixed(text[s_start:s_end], encoder, size)
                ]
    return _greedy_pack(text, units, encoder, size)


def sentence_packed(text: str, encoder, size: int) -> list[Span]:
    units = [s for start, end in _paragraphs(text) for s in _sentences(text, start, end)]
    return _greedy_pack(text, units, encoder, size)


def semantic(
    text: str,
    encoder,
    embed: Callable[[list[str]], Sequence],
    percentile: float,
    max_tokens: int = 1024,
) -> list[Span]:
    """Split where adjacent sentences are least similar, as LangChain's SemanticChunker does."""
    import numpy as np

    units = [s for start, end in _paragraphs(text) for s in _sentences(text, start, end)]
    if len(units) < 2:
        return units
    vectors = np.asarray(embed([text[a:b] for a, b in units]))
    similarity = (vectors[:-1] * vectors[1:]).sum(axis=1)
    cut_below = np.percentile(similarity, 100 - percentile)
    groups: list[list[Span]] = [[units[0]]]
    for i, unit in enumerate(units[1:]):
        if similarity[i] < cut_below:
            groups.append([])
        groups[-1].append(unit)
    spans: list[Span] = []
    for group in groups:
        spans += _greedy_pack(text, group, encoder, max_tokens)
    return spans
