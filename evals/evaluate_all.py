"""Run and save A, B, C, D sequentially, then generate a comparison."""
from datetime import datetime, timezone
from pathlib import Path

from evals.evaluate import parser, prepare, run_stage
from evals.report import METHODS, create_report


def main():
    args = parser().parse_args()
    corpus, rows, sha, validated = prepare(args)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    results = []
    for method in METHODS:
        # run_stage writes both stage files before the next retriever is created.
        results.append(run_stage(method, corpus, rows, sha, validated, args, run_id))
    create_report(results, Path(args.output_dir) / 'runs' / run_id)
    create_report(results, args.output_dir)
    print(f'Comparison saved to {Path(args.output_dir) / "comparison.md"}', flush=True)


if __name__ == '__main__':
    main()
