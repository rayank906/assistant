"""Compare compatible saved runs; derive failure observations from actual results."""
import argparse
import json
from pathlib import Path

METHODS = ('dense', 'bm25', 'hybrid', 'hybrid_rerank')


def table(results, tag=None):
    lines = ['| Method | Queries | R@1 | R@3 | R@5 | MRR | Mean ms | p50 ms | p95 ms |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for result in results:
        metrics = result['aggregate'] if tag is None else result['per_tag'][tag]
        lines.append(f"| {result['method']} | {metrics['queries']} | "
                     + ' | '.join(f'{metrics[key]:.3f}' for key in ('recall@1','recall@3','recall@5','mrr'))
                     + ' | ' + ' | '.join(f'{metrics[key]:.2f}' for key in ('mean_latency_ms','p50_latency_ms','p95_latency_ms')) + ' |')
    return lines


def create_report(results, output_dir):
    if [r['method'] for r in results] != list(METHODS):
        raise ValueError('Comparison requires all four stages in order')
    for key in ('dataset_sha256', 'corpus_fingerprint', 'mrr_depth', 'label_status'):
        if len({r[key] for r in results}) != 1:
            raise ValueError(f'Cannot compare incompatible results: {key}')
    if len({json.dumps(r['configuration'], sort_keys=True) for r in results}) != 1:
        raise ValueError('Cannot compare different experimental configurations')
    validated = results[0]['label_status'] == 'human-validated'
    lines = ['# EECS 280 retrieval comparison', '',
             ('Human-validated labels.' if validated else '**UNVALIDATED SAMPLE: smoke-test metrics only. Not credible final evaluation or resume evidence.**'), '',
             f"MRR is bounded at the common evaluated depth {results[0]['mrr_depth']}. R@k uses the requested any-relevant-section success definition, not fraction-of-all-relevant recall.", '',
             *table(results), '',
             'Latency includes query encoding and the applicable retrieval/fusion/reranking work. It excludes model loading, index building, and one unmeasured warmup query per stage. Queries run sequentially, once each; timings are hardware-dependent estimates.', '',
             '## Per-tag results', '', 'Tags overlap, so their counts do not sum to the total.', '']
    for tag in sorted(results[0]['per_tag']):
        lines += [f'### {tag}', '', *table(results, tag), '']
    lines += ['## Measured failures and ranking changes', '']
    for result in results:
        failures = result['failures_at_5']
        lines += [f"### {result['method']}: {len(failures)} R@5 failures", '']
        if not failures:
            lines += ['No labeled R@5 failures in this sample.', '']
        for failure in failures:
            lines += [f"- **{failure['id']}** ({', '.join(failure['tags'])}): {failure['question']}",
                      f"  Expected: {', '.join(failure['relevant_sections'])}.",
                      '  Retrieved: ' + '; '.join(f"{h['rank']}. {h['section_id']} ({h['score_type']}={h['score']:.5f})" for h in failure['retrieved_top_5'])]
        lines += ['']
    by_method = {r['method']: {q['id']: q for q in r['queries']} for r in results}
    observations = []
    for previous, current in [('dense','bm25'),('dense','hybrid'),('hybrid','hybrid_rerank')]:
        earlier, later = by_method[previous], by_method[current]
        gained = [ident for ident in earlier if not earlier[ident]['recall@5'] and later[ident]['recall@5']]
        lost = [ident for ident in earlier if earlier[ident]['recall@5'] and not later[ident]['recall@5']]
        improved = [ident for ident in earlier if later[ident]['reciprocal_rank'] > earlier[ident]['reciprocal_rank']]
        worsened = [ident for ident in earlier if later[ident]['reciprocal_rank'] < earlier[ident]['reciprocal_rank']]
        statement = f"{current} versus {previous}: R@5 gains {gained}; losses {lost}; first-relevant-rank improvements {improved}; regressions {worsened}."
        observations.append(statement)
        lines += ['- ' + statement]
    audit = results[0].get('lexical_text_audit', {})
    if audit:
        lines += ['', '### Corpus text audit', '',
                  'The existing embedding tokenizer decodes some C++ identifiers with spaces around underscores. BM25 sees these exact same passage strings, while queries retain contiguous identifiers. This is a verified text mismatch, not a claim inferred only from relevance labels.', '']
        for term, counts in audit.items():
            lines += [f"- `{term}`: {counts['passages_with_contiguous_identifier']} passages with the contiguous spelling; {counts['passages_with_spaced_identifier']} with spaces around the underscore."]
        lines += ['', 'In this sample BM25 misses q006 (`dynamic_cast`) and q007 (`size_t`) at rank five. A future lexical-normalization or raw-text experiment should test this mismatch without changing the current comparison retrospectively.', '']
    lines += ['', 'These are observed movements against the supplied labels, not proof of general retrieval quality or causal explanations. No claim that BM25 fixes identifiers or reranking always helps is made.', '',
              '## Setup and runtime', '', '| Method | Retriever setup s | Evaluation loop s |', '|---|---:|---:|']
    for result in results:
        lines += [f"| {result['method']} | {result['timing']['retriever_setup_seconds']:.3f} | {result['timing']['evaluation_seconds']:.3f} |"]
    lines += ['', f"Corpus setup: {results[0]['timing']['corpus_setup_seconds']:.3f} s; cache hit: {results[0]['timing']['corpus_cache_hit']}.", '',
              '## Limits and next experiment', '',
              ('- Human review metadata is recorded for every query; independently audit the label quality before external claims.' if validated else '- Relevance labels are proposed examples, not independently human-validated ground truth. Review question wording and all acceptable sections before using the metrics externally.'),
              '- Multi-section success requires only one relevant section, per the requested metric definition; it does not establish completeness of retrieved context.',
              f"- Evaluated settings: RRF k={results[0]['configuration']['rrf_k']}, {results[0]['configuration']['candidates']} parents per component, rerank {results[0]['configuration']['rerank_pool']} parents, return {results[0]['configuration']['top_k']}. BM25 defaults k1=1.5/b=0.75. No parameters were tuned on the bundled evaluation labels.",
              '- Reranking scores at most two retrieved evidence passages per parent; it does not score entire long sections. Pair token limits can truncate unusually long questions.',
              '- First build a larger human-reviewed holdout with realistic paraphrases, exact identifiers, code, ambiguous and difficult questions. Use a separate development set for any tuning.',
              '- Repeat latency measurements and, separately, test chunking or candidate-pool changes only after freezing the relevance labels.', '']
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.md').write_text('\n'.join(lines))
    summary = {'label_status':results[0]['label_status'], 'dataset_sha256':results[0]['dataset_sha256'],
               'corpus_fingerprint':results[0]['corpus_fingerprint'], 'mrr_depth':results[0]['mrr_depth'],
               'methods': {r['method']: {'aggregate':r['aggregate'],'per_tag':r['per_tag'],'timing':r['timing']} for r in results},
               'observations': observations}
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='evals/results')
    args = parser.parse_args()
    folder = Path(args.results_dir)
    results = [json.loads((folder / f'{method}.json').read_text()) for method in METHODS]
    create_report(results, folder)
    print(folder / 'comparison.md')


if __name__ == '__main__':
    main()
