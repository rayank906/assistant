"""Small inverted-index Okapi BM25 implementation using positive Lucene IDF.

Defaults k1=1.5, b=0.75. No stemming or stopword removal; C++ identifiers
and operator names are retained. Indexes the exact dense child passage text.
"""
from collections import Counter, defaultdict
import math
import re

import numpy as np

from retrievers.common import collapse_passages


def tokenize(text):
    # The embedding tokenizer may insert whitespace into operator names.
    text = re.sub(r'operator\s*([+*/%=!<>&|^~\-]+|\[\s*\]|\(\s*\))',
                  lambda m: 'operator' + re.sub(r'\s+', '', m.group(1)), text.lower())
    return re.findall(r'operator(?:\[\]|\(\)|[+*/%=!<>&|^~\-]+)|c\+\+|[a-z_][a-z_0-9]*|\d+', text)


class BM25Retriever:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('Require k1>0 and 0<=b<=1')
        self.corpus, self.k1, self.b = corpus, k1, b
        self.parameters = {"k1": k1, "b": b, "idf": "log(1+(N-df+0.5)/(df+0.5))", "tokenizer": "C++ aware regex; lowercased, no stemming or stopword removal"}
        postings = defaultdict(list)
        lengths = []
        for index, passage in enumerate(corpus.passages):
            tokens = tokenize(passage)
            lengths.append(len(tokens))
            for term, frequency in Counter(tokens).items():
                postings[term].append((index, frequency))
        n = len(lengths)
        self.norm = k1 * (1 - b + b * np.asarray(lengths) / max(float(np.mean(lengths)), 1))
        self.postings = {}
        for term, values in postings.items():
            ids, frequencies = zip(*values)
            idf = math.log1p((n - len(ids) + .5) / (len(ids) + .5))
            self.postings[term] = (np.asarray(ids), np.asarray(frequencies), idf)

    def retrieve(self, question, top_k=10):
        if top_k <= 0:
            return []
        scores = np.zeros(len(self.corpus.passages))
        for term, query_frequency in Counter(tokenize(question)).items():
            if term not in self.postings:
                continue
            ids, frequencies, idf = self.postings[term]
            scores[ids] += query_frequency * idf * frequencies * (self.k1 + 1) / (frequencies + self.norm[ids])
        # Zero-score documents are not lexical matches. Stable passage-ID tie break.
        ranked = np.argsort(-scores, kind='stable')
        ranked = ranked[scores[ranked] > 0]
        return collapse_passages(self.corpus, ranked, scores[ranked], top_k, 'bm25')
