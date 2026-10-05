# EECS 280 retrieval comparison

**Sample UNV to check**

MRR is bounded at the common evaluated depth 10. R@k uses the requested any-relevant-section success definition, not fraction-of-all-relevant recall.

| Method | Queries | R@1 | R@3 | R@5 | MRR | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 20 | 0.650 | 1.000 | 1.000 | 0.808 | 17.49 | 17.59 | 20.26 |
| bm25 | 20 | 0.550 | 0.850 | 0.850 | 0.699 | 0.19 | 0.19 | 0.30 |
| hybrid | 20 | 0.750 | 0.950 | 0.950 | 0.833 | 16.01 | 15.68 | 20.21 |
| hybrid_rerank | 20 | 0.750 | 1.000 | 1.000 | 0.858 | 2243.84 | 2204.28 | 2606.34 |

Latency includes query encoding and the applicable retrieval/fusion/reranking work. It excludes model loading, index building, and one unmeasured warmup query per stage. Queries run sequentially, once each; timings are hardware-dependent estimates.

## Per-tag results

Tags overlap, so their counts do not sum to the total.

### code

| Method | Queries | R@1 | R@3 | R@5 | MRR | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 6 | 0.833 | 1.000 | 1.000 | 0.917 | 16.84 | 16.86 | 19.58 |
| bm25 | 6 | 0.500 | 1.000 | 1.000 | 0.694 | 0.17 | 0.17 | 0.20 |
| hybrid | 6 | 0.833 | 1.000 | 1.000 | 0.889 | 16.44 | 15.89 | 19.33 |
| hybrid_rerank | 6 | 0.667 | 1.000 | 1.000 | 0.806 | 2292.29 | 2204.28 | 2539.74 |

### conceptual

| Method | Queries | R@1 | R@3 | R@5 | MRR | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 9 | 0.556 | 1.000 | 1.000 | 0.741 | 18.02 | 17.75 | 20.18 |
| bm25 | 9 | 0.778 | 0.889 | 0.889 | 0.852 | 0.23 | 0.21 | 0.31 |
| hybrid | 9 | 0.667 | 1.000 | 1.000 | 0.815 | 16.51 | 15.94 | 20.95 |
| hybrid_rerank | 9 | 0.667 | 1.000 | 1.000 | 0.815 | 2146.37 | 2147.15 | 2376.94 |

### exact-term

| Method | Queries | R@1 | R@3 | R@5 | MRR | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 6 | 0.833 | 1.000 | 1.000 | 0.917 | 17.47 | 17.63 | 20.20 |
| bm25 | 6 | 0.333 | 0.667 | 0.667 | 0.524 | 0.15 | 0.14 | 0.19 |
| hybrid | 6 | 0.833 | 0.833 | 0.833 | 0.833 | 14.42 | 14.34 | 17.04 |
| hybrid_rerank | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 2376.49 | 2364.61 | 2609.84 |

### multi-section

| Method | Queries | R@1 | R@3 | R@5 | MRR | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 4 | 0.250 | 1.000 | 1.000 | 0.583 | 17.74 | 17.15 | 19.56 |
| bm25 | 4 | 0.750 | 1.000 | 1.000 | 0.875 | 0.24 | 0.24 | 0.30 |
| hybrid | 4 | 0.750 | 1.000 | 1.000 | 0.875 | 18.66 | 18.02 | 22.02 |
| hybrid_rerank | 4 | 0.750 | 1.000 | 1.000 | 0.833 | 2160.87 | 2154.77 | 2381.97 |

### syntax

| Method | Queries | R@1 | R@3 | R@5 | MRR | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 5 | 0.400 | 1.000 | 1.000 | 0.700 | 17.58 | 18.02 | 20.18 |
| bm25 | 5 | 0.200 | 1.000 | 1.000 | 0.533 | 0.19 | 0.19 | 0.21 |
| hybrid | 5 | 0.800 | 1.000 | 1.000 | 0.867 | 16.80 | 16.10 | 19.59 |
| hybrid_rerank | 5 | 0.600 | 1.000 | 1.000 | 0.767 | 2248.85 | 2199.59 | 2507.76 |

## Measured failures and ranking changes

### dense: 0 R@5 failures

No labeled R@5 failures in this sample.


### bm25: 3 R@5 failures

- **q003** (conceptual): Why hide the representation of an abstract data type?
  Expected: 280:eecs280notes.pdf:chapter-16:16.6.
  Retrieved: 1. 280:eecs280notes.pdf:chapter-11:11.1 (bm25=14.57897); 2. 280:eecs280notes.pdf:chapter-32:intro (bm25=12.08872); 3. 280:eecs280notes.pdf:chapter-11:11.4 (bm25=9.98107); 4. 280:eecs280notes.pdf:chapter-11:11.2 (bm25=9.95581); 5. 280:eecs280notes.pdf:chapter-16:16.1 (bm25=9.78387)
- **q006** (exact-term): What does dynamic_cast do?
  Expected: 280:eecs280notes.pdf:chapter-18:18.4.
  Retrieved: 1. 280:eecs280notes.pdf:chapter-5:intro (bm25=8.94711); 2. 280:eecs280notes.pdf:chapter-5:5.3 (bm25=7.33073); 3. 280:eecs280notes.pdf:chapter-35:35.5 (bm25=6.58984); 4. 280:eecs280notes.pdf:chapter-29:29.3 (bm25=6.22365); 5. 280:eecs280notes.pdf:chapter-22:22.6 (bm25=6.12108)
- **q007** (exact-term): What is size_t used for?
  Expected: 280:eecs280notes.pdf:chapter-22:22.6.
  Retrieved: 1. 280:eecs280notes.pdf:chapter-35:intro (bm25=5.92867); 2. 280:eecs280notes.pdf:chapter-5:intro (bm25=5.76843); 3. 280:eecs280notes.pdf:chapter-27:intro (bm25=5.71080); 4. 280:eecs280notes.pdf:chapter-35:35.4 (bm25=5.66612); 5. 280:eecs280notes.pdf:chapter-2:intro (bm25=5.30880)

### hybrid: 1 R@5 failures

- **q006** (exact-term): What does dynamic_cast do?
  Expected: 280:eecs280notes.pdf:chapter-18:18.4.
  Retrieved: 1. 280:eecs280notes.pdf:chapter-26:26.6 (rrf=0.02715); 2. 280:eecs280notes.pdf:chapter-7:intro (rrf=0.02652); 3. 280:eecs280notes.pdf:chapter-38:intro (rrf=0.02614); 4. 280:eecs280notes.pdf:chapter-29:29.3 (rrf=0.02563); 5. 280:eecs280notes.pdf:chapter-18:18.2 (rrf=0.02558)

### hybrid_rerank: 0 R@5 failures

No labeled R@5 failures in this sample.


- bm25 versus dense: R@5 gains []; losses ['q003', 'q006', 'q007']; first-relevant-rank improvements ['q004', 'q005', 'q017', 'q020']; regressions ['q003', 'q006', 'q007', 'q010', 'q011', 'q015'].
- hybrid versus dense: R@5 gains []; losses ['q006']; first-relevant-rank improvements ['q005', 'q009', 'q016', 'q017']; regressions ['q003', 'q006', 'q011'].
- hybrid_rerank versus hybrid: R@5 gains ['q006']; losses []; first-relevant-rank improvements ['q003', 'q004', 'q006', 'q020']; regressions ['q005', 'q015', 'q017'].

### Corpus text audit

The existing embedding tokenizer decodes some C++ identifiers with spaces around underscores. BM25 sees these exact same passage strings, while queries retain contiguous identifiers. This is a verified text mismatch, not a claim inferred only from relevance labels.

- `dynamic_cast`: 0 passages with the contiguous spelling; 3 with spaces around the underscore.
- `size_t`: 0 passages with the contiguous spelling; 20 with spaces around the underscore.

In this sample BM25 misses q006 (`dynamic_cast`) and q007 (`size_t`) at rank five. A future lexical-normalization or raw-text experiment should test this mismatch without changing the current comparison retrospectively.


These are observed movements against the supplied labels, not proof of general retrieval quality or causal explanations. No claim that BM25 fixes identifiers or reranking always helps is made.

## Setup and runtime

| Method | Retriever setup s | Evaluation loop s |
|---|---:|---:|
| dense | 0.001 | 0.361 |
| bm25 | 0.103 | 0.011 |
| hybrid | 0.105 | 0.331 |
| hybrid_rerank | 0.349 | 44.889 |

Corpus setup: 6.435 s; cache hit: True.

## Limits and next experiment

- Relevance labels are proposed examples, not independently human-validated ground truth. Review question wording and all acceptable sections before using the metrics externally.
- Multi-section success requires only one relevant section, per the requested metric definition; it does not establish completeness of retrieved context.
- Evaluated settings: RRF k=60, 50 parents per component, rerank 20 parents, return 10. BM25 defaults k1=1.5/b=0.75. No parameters were tuned on the bundled evaluation labels.
- Reranking scores at most two retrieved evidence passages per parent; it does not score entire long sections. Pair token limits can truncate unusually long questions.
- First build a larger human-reviewed holdout with realistic paraphrases, exact identifiers, code, ambiguous and difficult questions. Use a separate development set for any tuning.
- Repeat latency measurements and, separately, test chunking or candidate-pool changes only after freezing the relevance labels.
