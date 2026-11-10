# chunking-bench

Chunking strategies for retrieval, benchmarked on SQuAD with answer-span ground truth. Every strategy is judged twice: at fixed k, and within a fixed token budget, because fixed k quietly gives larger chunks more context to work with.

Companion to [Your Chunking Benchmark Is Measuring Chunk Size](https://cloudandsre.com/blog/rag-chunking-strategies-measured/).

## Method

SQuAD v1.1 dev: 48 Wikipedia articles rebuilt as whole documents, 2,000 sampled questions whose answer offsets are known. A retrieved chunk counts as a hit only if it contains the entire answer span. Embeddings are `BAAI/bge-small-en-v1.5` through fastembed; token counts use `cl100k_base`.

Strategies in `chunkers.py`: fixed-size by tokens (with and without overlap), recursive (paragraphs, then sentences, then hard cuts), sentence-packed, semantic (split where adjacent sentence embeddings are least similar), and whole paragraphs as the gold-boundary reference.

## Results

| Strategy | Chunks | Mean tokens | Answers split | hit@1 | hit@5 | hit@10 | Within 1,000 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-128 | 2,631 | 127 | 1.6% | 50.5% | 78.8% | 86.7% | 83.2% |
| fixed-256 | 1,330 | 251 | 1.0% | 49.0% | 79.7% | 87.5% | 71.9% |
| fixed-512 | 676 | 494 | 0.7% | 44.9% | 78.4% | 87.8% | 46.1% |
| fixed-1024 | 350 | 954 | 0.2% | 39.3% | 73.0% | 85.3% | 3.9% |
| fixed-256-overlap15% | 1,545 | 253 | 0.0% | 50.6% | 80.7% | 89.6% | 74.0% |
| fixed-512-overlap15% | 780 | 499 | 0.0% | 45.8% | 80.3% | 89.8% | 47.5% |
| recursive-128 | 3,222 | 104 | 0.2% | 65.0% | 86.5% | 91.6% | 90.9% |
| recursive-256 | 1,784 | 187 | 0.0% | 62.5% | 87.3% | 92.0% | 86.9% |
| recursive-512 | 802 | 416 | 0.0% | 51.7% | 82.2% | 89.8% | 66.3% |
| sentence-256 | 1,440 | 232 | 0.2% | 55.0% | 82.5% | 89.2% | 80.5% |
| sentence-512 | 701 | 476 | 0.0% | 47.7% | 79.8% | 88.6% | 58.3% |
| semantic-p90 | 1,175 | 284 | 0.2% | 44.9% | 74.5% | 83.4% | 53.6% |
| semantic-p95 | 729 | 458 | 0.2% | 42.0% | 72.2% | 82.5% | 37.1% |
| paragraph (gold boundary) | 2,067 | 161 | 0.0% | 68.7% | 89.7% | 93.5% | 90.1% |

"Answers split" is the share of questions whose answer no single chunk contains. Semantic chunking took roughly 300 s of wall-clock to chunk the corpus against about a second for every other strategy; the machine was busy during the run, so read the ratio rather than the absolute. Full numbers in [`results/chunking.json`](results/chunking.json).

## Run it

```bash
pip install -r requirements-ci.txt
pytest -q
python bench.py            # downloads SQuAD dev (about 5 MB) and the embedding model
```

The full run embeds the corpus once per strategy and takes a while on a laptop.

## Limits

One dataset, one embedding model, short factoid questions over encyclopedia prose. Results on long technical documents, code or tables will differ. The benchmark measures whether the answer is retrievable, not whether a model answers correctly from the chunk.

## License

Apache-2.0
