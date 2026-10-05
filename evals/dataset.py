import hashlib
import json
from pathlib import Path

TAGS = {'conceptual', 'exact-term', 'code', 'syntax', 'multi-section'}


def load_dataset(path, section_ids, allow_unvalidated=False):
    data = Path(path).read_bytes()
    rows = json.loads(data)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Dataset must be a nonempty JSON array')
    seen = set()
    valid_ids = set(section_ids)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each query must be an object')
        for field in ('id', 'question'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Missing or invalid {field}')
        if row['id'] in seen:
            raise ValueError(f'Duplicate query ID: {row["id"]}')
        seen.add(row['id'])
        for field in ('relevant_sections', 'tags'):
            if not isinstance(row.get(field), list) or not row[field] or any(not isinstance(v, str) for v in row[field]):
                raise ValueError(f'{row["id"]}: invalid {field}')
            if len(row[field]) != len(set(row[field])):
                raise ValueError(f'{row["id"]}: duplicate {field}')
        if not isinstance(row.get('validation', {}), dict):
            raise ValueError(f'{row["id"]}: validation metadata must be an object')
        unknown = set(row['relevant_sections']) - valid_ids
        if unknown:
            raise ValueError(f'{row["id"]}: unknown sections: {sorted(unknown)}')
        if set(row['tags']) - TAGS:
            raise ValueError(f'{row["id"]}: unsupported tags')
    validated = all(row.get('validation', {}).get('status') == 'human-validated'
                    and isinstance(row['validation'].get('reviewer'), str)
                    and row['validation']['reviewer'].strip()
                    and isinstance(row['validation'].get('reviewed_at'), str)
                    and row['validation']['reviewed_at'].strip() for row in rows)
    if not validated and not allow_unvalidated:
        raise ValueError('Relevance labels need human validation. For a provisional smoke test only, pass --allow-unvalidated.')
    return rows, hashlib.sha256(data).hexdigest(), validated
