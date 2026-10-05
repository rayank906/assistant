from retrievers.bm25 import BM25Retriever
from retrievers.common import Hit
from retrievers.dense import DenseRetriever


class HybridRetriever:
    """Fuse deduplicated parent ranks; raw component scores are never added."""
    def __init__(self, corpus, candidates=50, rrf_k=60, dense=None, bm25=None):
        if candidates <= 0 or rrf_k <= 0:
            raise ValueError('candidates and rrf_k must be positive')
        self.corpus, self.candidates, self.rrf_k = corpus, candidates, rrf_k
        self.dense = dense or DenseRetriever(corpus)
        self.bm25 = bm25 or BM25Retriever(corpus)
        self.parameters = {'candidates_per_component': candidates, 'rrf_k': rrf_k,
                           'fusion_unit': 'deduplicated parent section', 'bm25': self.bm25.parameters}

    def retrieve(self, question, top_k=10):
        if top_k <= 0:
            return []
        if top_k > self.candidates:
            raise ValueError('top_k exceeds configured component candidate count')
        fused = {}
        for method, retriever in [('dense', self.dense), ('bm25', self.bm25)]:
            for rank, hit in enumerate(retriever.retrieve(question, self.candidates), 1):
                if hit.section_id not in fused:
                    fused[hit.section_id] = Hit(hit.section_id, hit.parent_index, hit.passage_index, 0., 'rrf', [], {})
                result = fused[hit.section_id]
                result.score += 1 / (self.rrf_k + rank)
                result.evidence = list(dict.fromkeys(result.evidence + hit.evidence))
                result.components[method] = {'rank': rank, 'score': hit.score, 'score_type': hit.score_type,
                                             'passage_index': hit.passage_index}
        return sorted(fused.values(), key=lambda h: (-h.score, h.section_id))[:top_k]
