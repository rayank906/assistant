from retrievers.common import collapse_passages


class DenseRetriever:
    """Same encoder, FAISS IP search, and adaptive parent dedup as the app."""
    def __init__(self, corpus):
        self.corpus = corpus

    def retrieve(self, question, top_k=10):
        if top_k <= 0:
            return []
        corpus = self.corpus
        embedding = corpus.model.encode([question], normalize_embeddings=True).astype('float32')
        count = min(max(top_k, 1), corpus.index.ntotal)
        while count:
            scores, indices = corpus.index.search(embedding, count)
            hits = collapse_passages(corpus, indices[0], scores[0], top_k, 'cosine_similarity')
            if len(hits) >= top_k or count == corpus.index.ntotal:
                return hits
            count = min(count * 2, corpus.index.ntotal)
        return []
