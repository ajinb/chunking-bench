import numpy as np
import pytest
import tiktoken

import bench
import chunkers

ENC = tiktoken.get_encoding("cl100k_base")
DOC = (
    "Alpha is first. Beta comes next! Is gamma third?\n\n"
    "A second paragraph lives here. It has two sentences.\n\n" + "Word " * 400
)


def covered(text, spans):
    return "".join(text[a:b] for a, b in spans)


@pytest.mark.parametrize("size", [16, 64, 256])
def test_fixed_covers_the_document_without_gaps(size):
    spans = chunkers.fixed(DOC, ENC, size)
    assert covered(DOC, spans) == DOC
    assert all(a == b for (_, a), (b, _) in zip(spans, spans[1:], strict=False))


def test_fixed_respects_size():
    for a, b in chunkers.fixed(DOC, ENC, 32):
        assert len(ENC.encode(DOC[a:b])) <= 36


def test_overlap_repeats_text():
    plain, lapped = chunkers.fixed(DOC, ENC, 64), chunkers.fixed(DOC, ENC, 64, 16)
    assert len(lapped) > len(plain)
    assert lapped[1][0] < lapped[0][1]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunkers.fixed(DOC, ENC, 16, 16)


def test_paragraphs_split_on_blank_lines():
    spans = chunkers.paragraphs(DOC)
    assert DOC[spans[0][0] : spans[0][1]].endswith("third?")
    assert len(spans) == 3


@pytest.mark.parametrize("fn", [chunkers.recursive, chunkers.sentence_packed])
def test_packed_strategies_keep_order_and_never_overlap(fn):
    spans = fn(DOC, ENC, 40)
    assert all(a < b <= c < d for (a, b), (c, d) in zip(spans, spans[1:], strict=False))


def test_recursive_keeps_a_small_paragraph_whole():
    spans = chunkers.recursive(DOC, ENC, 200)
    assert (
        DOC[spans[0][0] : spans[0][1]].startswith("Alpha")
        and "third?" in DOC[spans[0][0] : spans[0][1]]
    )


def test_recursive_hard_cuts_a_giant_sentence():
    for a, b in chunkers.recursive(DOC, ENC, 40):
        assert len(ENC.encode(DOC[a:b])) <= 44


def test_semantic_splits_at_the_low_similarity_seam():
    text = "cat one. cat two. dog one. dog two."
    vectors = {"cat one.": [1, 0], "cat two.": [1, 0], "dog one.": [0, 1], "dog two.": [0, 1]}
    spans = chunkers.semantic(
        text, ENC, lambda items: np.array([vectors[i] for i in items], float), percentile=40
    )
    assert [text[a:b] for a, b in spans] == ["cat one. cat two.", "dog one. dog two."]


def test_answer_must_be_entirely_inside_the_span():
    assert bench.contains_answer((10, 20), [(12, 18)])
    assert not bench.contains_answer((10, 20), [(15, 25)])
    assert bench.contains_answer((10, 20), [(0, 5), (11, 12)])
