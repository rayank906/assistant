import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from evals.dataset import load_dataset
from evals.metrics import aggregate, per_tag, query_metrics
from retrievers.corpus import ROOT, digest, export_catalog, load_corpus

METHODS = {'dense': ('retrievers.dense', 'DenseRetriever'),
           'bm25': ('retrievers.bm25', 'BM25Retriever'),
           'hybrid': ('retrievers.hybrid', 'HybridRetriever'),
           'hybrid_rerank': ('retrievers.reranker', 'RerankingRetriever')}


def make_retriever(method, corpus, args):
    module, name = METHODS[method]
    cls = getattr(importlib.import_module(module), name)
    if method == "hybrid_rerank":
        return cls(corpus, candidates=args.candidates, rrf_k=args.rrf_k,
                   pool_size=args.rerank_pool, model_name=args.reranker_model)
    if method == "hybrid":
        return cls(corpus, candidates=args.candidates, rrf_k=args.rrf_k)
    return cls(corpus)


def save_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history = path.parent / 'history'
        history.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        # Preserve every previous result; only the latest alias is replaced.
        (history / f'{path.stem}-{stamp}.json').write_bytes(path.read_bytes())
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def run_stage(method, corpus, rows, dataset_sha, validated, args, run_id):
    setup_start = time.perf_counter()
    retriever = make_retriever(method, corpus, args)
    setup_seconds = time.perf_counter() - setup_start
    # Unmeasured warmup uses no eval labels. Query embedding is NOT cached.
    retriever.retrieve('Explain the course notes.', args.top_k)
    results = []
    measured_start = time.perf_counter()
    for row in rows:
        start = time.perf_counter()
        hits = retriever.retrieve(row['question'], args.top_k)
        elapsed = (time.perf_counter() - start) * 1000
        retrieved = []
        for rank, hit in enumerate(hits, 1):
            chunk = corpus.chunks[hit.parent_index]
            retrieved.append({
                'rank': rank, **asdict(hit), 'chapter': chunk.chapter,
                'section': chunk.section, 'source_file': chunk.source_file,
                'page_start': chunk.page_start, 'page_end': chunk.page_end,
                'passage_text': corpus.passages[hit.passage_index],
            })
        results.append({**row, 'retrieved': retrieved, 'latency_ms': elapsed,
                        **query_metrics([hit.section_id for hit in hits], row['relevant_sections'])})
    result = {
        'method': method, 'run_id': run_id,
        'implementation_sha256': {str(path.relative_to(ROOT)): digest(path)
                                  for folder in ('retrievers', 'evals')
                                  for path in sorted((ROOT / folder).glob('*.py'))},
        'label_status': 'human-validated' if validated else 'UNVALIDATED SAMPLE — NOT FINAL EVIDENCE',
        'dataset_sha256': dataset_sha, 'corpus_fingerprint': corpus.fingerprint,
        'corpus': {**corpus.manifest, 'sections': len(corpus.chunks), 'passages': len(corpus.passages)},
        'configuration': {key: getattr(args, key) for key in ('top_k', 'candidates', 'rrf_k', 'rerank_pool', 'reranker_model')},
        'method_parameters': getattr(retriever, 'parameters', {}),
        'lexical_text_audit': {term: {
            'passages_with_contiguous_identifier': sum(term in p for p in corpus.passages),
            'passages_with_spaced_identifier': sum(term.replace('_', ' _ ') in p for p in corpus.passages),
        } for term in ('dynamic_cast', 'size_t')},
        'mrr_depth': args.top_k, 'recall_definition': 'at least one relevant parent section in top k (hit rate)',
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'device': str(corpus.model.device), 'torch_threads': __import__('torch').get_num_threads()},
        'timing': {'corpus_setup_seconds': corpus.setup_seconds, 'corpus_cache_hit': corpus.cache_hit,
                   'retriever_setup_seconds': setup_seconds, 'evaluation_seconds': time.perf_counter() - measured_start,
                   'warmup_queries': 1, 'query_latency_includes': 'query encoding, candidate retrieval, fusion and reranking as applicable; excludes loading/index construction'},
        'aggregate': aggregate(results), 'per_tag': per_tag(results), 'queries': results,
        'failures_at_5': [{key: row[key] for key in ('id', 'question', 'relevant_sections', 'tags')}
                          | {'retrieved_top_5': row['retrieved'][:5]} for row in results if not row['recall@5']],
    }
    save_result(Path(args.output_dir) / f'{method}.json', result)
    save_result(Path(args.output_dir) / 'runs' / run_id / f'{method}.json', result)
    metrics = result['aggregate']
    print(f"{method} [{result['label_status']}]", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print('Per-tag metrics: ' + json.dumps(result['per_tag'], sort_keys=True), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description='Parent-section retrieval evaluation (no answer model calls)')
    p.add_argument('--method', choices=list(METHODS), default='dense')
    p.add_argument('--dataset', default=str(ROOT / 'evals/eecs280_questions.json'))
    p.add_argument('--pdf', default=str(ROOT / 'eecs280/eecs280notes.pdf'))
    p.add_argument('--cache-dir', default=str(ROOT / '.cache/retrieval_eval'))
    p.add_argument('--output-dir', default=str(ROOT / 'evals/results'))
    p.add_argument('--allow-unvalidated', action='store_true')
    p.add_argument('--top-k', type=int, default=10, help='Evaluation result depth; MRR is bounded at this depth')
    p.add_argument('--candidates', type=int, default=50, help='Unique parents per component for RRF')
    p.add_argument('--rrf-k', type=int, default=60)
    p.add_argument('--rerank-pool', type=int, default=20)
    p.add_argument('--reranker-model', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    p.add_argument('--threads', type=int, default=2)
    return p


def prepare(args):
    if args.top_k < 5 or args.candidates < max(args.top_k, args.rerank_pool) or args.rerank_pool < args.top_k or args.rrf_k <= 0 or args.threads < 1:
        raise ValueError('Require top-k >=5, candidates/rerank-pool >=top-k, rrf-k/threads >0')
    import torch
    torch.set_num_threads(args.threads)
    corpus = load_corpus(args.pdf, args.cache_dir)
    rows, sha, validated = load_dataset(args.dataset, corpus.section_ids, args.allow_unvalidated)
    export_catalog(corpus, ROOT / 'evals/section_catalog.json')
    return corpus, rows, sha, validated


def main():
    args = parser().parse_args()
    corpus, rows, sha, validated = prepare(args)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_stage(args.method, corpus, rows, sha, validated, args, run_id)


if __name__ == '__main__':
    main()
