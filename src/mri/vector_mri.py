import math
from collections import Counter
from typing import List, Tuple

from src.corpus.corpus import Corpus
from src.corpus.query import Query
from src.mri.mri import MRI


class VectorMRI(MRI):

    def __init__(self, corpus: Corpus, smoothing: int = 0.4):
        super().__init__(corpus)
        self.smoothing = smoothing
        self.tf = {}
        self.idf = {}
        self.process_corpus()

    def query(self, query: Query) -> List[Tuple[str, float]]:
        
        assert len(query.text) != 0
        
        w = self.process_query(query)
        
        docs = []
        for doc in self.corpus.documents.keys():
            docs.append((doc, self.sim(self.vec_doc(doc), w)))
            
        docs.sort(key=lambda x: x[1], reverse=True)
        
        return docs

    def process_corpus(self) -> None:
        self.tf = self.calculate_tf()
        self.idf = self.calculate_idf()
        
    def process_query(self, query: Query) -> List[float]:
        w = []
        N = len(self.corpus.documents.keys())
        counter = Counter(query)
        max_count = counter.most_common(1)[0][1]
        
        for word in self.corpus.vocabulary.keys():
            if counter[word] == 0:
                w.append(0)
                continue
            
            n_i = self.corpus.vocabulary[word]
            w_i = (self.smoothing + (1 - self.smoothing)*counter[word]/max_count) * math.log(N/n_i)
            w.append(w_i)
        
        return w

    def calculate_tf(self):
        tf = {}
    
        for doc, counter in self.corpus.documents.items():
            max_count = counter.most_common(1)[0][1]
        
            for word in self.corpus.vocabulary.keys():
                tf[word, doc] = (counter[word] / max_count)

        return tf
    
    def calculate_idf(self):
        idf = {}
        N = len(self.corpus.documents.keys())
        
        for word, n_i in self.corpus.vocabulary.items():
            idf[word] = math.log(N / n_i)
        
        return idf
    
    def w(self, word: str, doc: str):
        return self.tf[word, doc] * self.idf[word]

    def vec_doc(self, doc: str):
        doc_w = []
        for word in self.corpus.vocabulary.keys():
            doc_w.append(self.w(word, doc))
        return doc_w

    def sim(self, doc_w: List[float], query_w: List[float]):
        
        assert len(doc_w) == len(query_w)
    
        numerator = sum(map(lambda x: x[0] * x[1], zip(doc_w, query_w)))
        
        denominator = self.norm_2(doc_w) * self.norm_2(query_w)
        
        assert denominator != 0
        
        return numerator/denominator

    @staticmethod
    def norm_2(w: List[float]):
        return math.sqrt(sum(map(lambda x: x**2, w)))
