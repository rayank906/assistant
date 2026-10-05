"""Persist the existing parser's exact passages and normalized dense embeddings."""
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import faiss
import numpy as np

from retrievers.common import section_id

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Corpus:
    chunks: list
    passages: list
    parents: list
    ranges: list
    section_ids: list
    index: object
    model: object
    fingerprint: str
    manifest: dict
    cache_hit: bool
    setup_seconds: float


def load_corpus(pdf_path=None, cache_dir=None):
    start = time.perf_counter()
    from block_chunk import model, extract_section_chunks, create_search_passages, create_embeddings, SectionChunk
    pdf = Path(pdf_path or ROOT / 'eecs280/eecs280notes.pdf').resolve()
    manifest = {
        'pdf_sha256': digest(pdf), 'parser_sha256': digest(ROOT / 'block_chunk.py'),
        'model': 'sentence-transformers/all-MiniLM-L6-v2',
        'model_revision': getattr(model[0].auto_model.config, '_commit_hash', None),
        'max_seq_length': model.max_seq_length, 'normalize_embeddings': True,
        'versions': {name: importlib.metadata.version(name) for name in ('sentence-transformers', 'transformers', 'torch', 'faiss-cpu', 'numpy', 'PyMuPDF')},
    }
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    folder = Path(cache_dir or ROOT / '.cache/retrieval_eval') / fingerprint
    metadata_path = folder / 'corpus.json'
    index_path = folder / 'index.faiss'
    embedding_path = folder / 'embeddings.npy'
    cached = metadata_path.exists() and index_path.exists() and embedding_path.exists()
    if cached:
        data = json.loads(metadata_path.read_text())
        chunks = [SectionChunk(**chunk) for chunk in data['chunks']]
        passages, parents, ranges = data['passages'], data['parents'], data['ranges']
        if data['manifest'] != manifest:
            raise ValueError('Corpus cache manifest mismatch')
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(passages):
            raise ValueError('Corpus cache index size mismatch')
    else:
        chunks = extract_section_chunks(str(pdf))
        passages, parents, ranges = create_search_passages(chunks, include_ranges=True)
        index, embeddings = create_embeddings(passages)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, embeddings)
        faiss.write_index(index, str(index_path))
        metadata_path.write_text(json.dumps({
            'manifest': manifest, 'chunks': [asdict(c) for c in chunks],
            'passages': passages, 'parents': parents, 'ranges': ranges,
        }, ensure_ascii=False))
    ids = [section_id(c) for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError('Parent section identifiers must be unique')
    return Corpus(chunks, passages, parents, ranges, ids, index, model,
                  fingerprint, manifest, cached, time.perf_counter() - start)


def export_catalog(corpus, path):
    Path(path).write_text(json.dumps([
        {'section_id': ident, **asdict(chunk)}
        for ident, chunk in zip(corpus.section_ids, corpus.chunks)
    ], indent=2, ensure_ascii=False) + '\n')
