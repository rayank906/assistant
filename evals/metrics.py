import numpy as np


def query_metrics(retrieved_ids, relevant_ids):
    # Defense in depth: repeated child hits must never inflate parent ranks.
    ids = list(dict.fromkeys(retrieved_ids))
    relevant = set(relevant_ids)
    first = next((rank for rank, ident in enumerate(ids, 1) if ident in relevant), None)
    return {
        'first_relevant_rank': first,
        'recall@1': int(first is not None and first <= 1),
        'recall@3': int(first is not None and first <= 3),
        'recall@5': int(first is not None and first <= 5),
        'reciprocal_rank': 1 / first if first else 0.0,
    }


def aggregate(rows):
    if not rows:
        return {'queries': 0}
    latencies = [r['latency_ms'] for r in rows]
    return {
        'queries': len(rows),
        **{key: float(np.mean([r[key] for r in rows])) for key in ('recall@1', 'recall@3', 'recall@5')},
        'mrr': float(np.mean([r['reciprocal_rank'] for r in rows])),
        'mean_latency_ms': float(np.mean(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
    }


def per_tag(rows):
    return {tag: aggregate([r for r in rows if tag in r['tags']])
            for tag in sorted({tag for row in rows for tag in row['tags']})}
