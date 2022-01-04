import math
import nltk
from typing import List, Tuple

from src.corpus.corpus import Corpus
from src.corpus.query import Query
from src.mri.mri import MRI


class VectorMRI(MRI):

    def __init__(self, corpus: Corpus, smoothing: int = 0.4):
        super().__init__(corpus)
        self.smoothing = smoothing
        # self.process_corpus()

    def query(self, query: Query) -> List[Tuple[str, float]]:
        
        if self.corpus.stemmer:
            words = [self.corpus.stemmer.stem(w) for w in query.words]
        else:
            words = query.words
        query_counter = nltk.Counter(words)
        ranking = []
        
        for id, doc in self.corpus.documents.items():
    
            num = 0
            doc_weights_sqr = 0
            query_weights_sqr = 0
            for word in query_counter.keys():
                if word not in self.corpus.vocabulary:
                    continue
                w_doc = self.weight_doc(word, id)
                w_query = self.weight_query(word, query_counter)
                num += w_doc * w_query
                doc_weights_sqr += w_doc ** 2
                query_weights_sqr += w_query ** 2

            try:
                sim = num / (math.sqrt(doc_weights_sqr) * math.sqrt(query_weights_sqr))
            except ZeroDivisionError:
                sim = 0
            
            if sim > 0.3:
                ranking.append((id, sim))

        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking

    def weight_query(self, ti: str, query_vect: nltk.Counter):
        freq = query_vect[ti]
        max_freq = query_vect.most_common(1)[0][1]
        tf = freq / max_freq
        idf = self.idf(ti)
        return (self.smoothing + (1 - self.smoothing) * tf) * idf

    def weight_doc(self, ti: str, dj: str) -> float:
        return self.tf(ti, dj) * self.idf(ti)

    def tf(self, ti: str, dj: str) -> float:
        freq = self.corpus.get_frequency(ti, dj)
        max_freq_tok, max_freq = self.corpus.get_max_frequency(dj)
        return freq / max_freq

    def idf(self, ti: str) -> float:
        N = len(self.corpus.documents)
        ni = self.corpus.vocabulary[ti]
        return math.log(N / ni)
