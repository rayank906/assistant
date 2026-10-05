from pathlib import Path

from retrievers.common import Hit
from retrievers.hybrid import HybridRetriever


class RerankingRetriever:
    """Rerank evidence from the hybrid top parent pool, never the whole corpus."""
    def __init__(self, corpus, candidates=50, rrf_k=60, pool_size=20,
                 model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', encoder=None):
        if pool_size <= 0 or pool_size > candidates:
            raise ValueError('Require 0 < pool_size <= candidates')
        self.corpus, self.pool_size = corpus, pool_size
        self.hybrid = HybridRetriever(corpus, candidates, rrf_k)
        if encoder is None:
            from sentence_transformers import CrossEncoder
            import torch
            self.encoder = CrossEncoder(model_name, device=str(corpus.model.device),
                                        max_length=512, activation_fn=torch.nn.Identity(),
                                        cache_folder=str(Path(__file__).resolve().parents[1] / '.cache/retrieval_models'))
        else:
            self.encoder = encoder
        config = getattr(getattr(self.encoder, 'model', None), 'config', None)
        self.parameters = {
            **self.hybrid.parameters, 'rerank_parent_pool': pool_size,
            'model': model_name, 'model_revision': getattr(config, '_commit_hash', None),
            'max_length': 512, 'activation': 'identity (raw logit)',
            'parent_score': 'max cross-encoder score over best dense/BM25 child evidence',
            'max_pairs_per_query': pool_size * 2,
        }

    def retrieve(self, question, top_k=10):
        if top_k <= 0:
            return []
        if top_k > self.pool_size:
            raise ValueError('top_k exceeds reranker candidate pool')
        candidates = self.hybrid.retrieve(question, self.pool_size)
        if not candidates:
            return []
        pairs, owners = [], []
        for parent_rank, hit in enumerate(candidates):
            for passage in hit.evidence:
                pairs.append((question, self.corpus.passages[passage]))
                owners.append((parent_rank, passage))
        scores = self.encoder.predict(pairs, batch_size=16, show_progress_bar=False)
        best = {}
        for (parent_rank, passage), score in zip(owners, scores):
            score = float(score)
            if parent_rank not in best or score > best[parent_rank][0]:
                best[parent_rank] = (score, passage)
        results = []
        for parent_rank, hit in enumerate(candidates):
            score, passage = best[parent_rank]
            results.append(Hit(hit.section_id, hit.parent_index, passage, score, 'cross_encoder_logit',
                               hit.evidence, {**hit.components, 'hybrid': {'rank': parent_rank + 1, 'score': hit.score}}))
        return sorted(results, key=lambda h: (-h.score, h.section_id))[:top_k]
