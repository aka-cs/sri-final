import json
import math
import os
from functools import reduce

import nltk
from typing import List, Tuple

from src.corpus.corpus import Corpus, Document
from src.corpus.query import Query
from src.mri.mri import MRI


class VectorMRI(MRI):
    
    def __init__(self, corpus: Corpus, smoothing: int = 0.4):
        super().__init__(corpus)
        self.smoothing = smoothing
        self.processed_queries = {}
        # self.process_corpus()
    
    def query(self, query: Query) -> List[Tuple[Document, float]]:
        
        words = self.stem_query(query)
        words = [word for word in words if word in self.corpus.vocabulary]
        query_counter = nltk.Counter(words)
        self.processed_queries[query] = query_counter
        ranking = []
        
        q_vector = None
        if self.feedback_path:
            q_vector = self.load_query(query)
        if q_vector is None:
            q_vector = [self.weight_query(word, query_counter) for word in query_counter.keys()]
        
        for doc, counter in self.corpus.documents.items():
            
            num = 0
            doc_weights_sqr = 0
            query_weights_sqr = 0
            for i, word in enumerate(query_counter.keys()):
                w_doc = self.weight_doc(word, doc)
                w_query = q_vector[i]
                num += w_doc * w_query
                doc_weights_sqr += w_doc ** 2
                query_weights_sqr += w_query ** 2
            
            try:
                sim = num / (math.sqrt(doc_weights_sqr) * math.sqrt(query_weights_sqr))
            except ZeroDivisionError:
                sim = 0
            
            if sim > 0.4:
                ranking.append((doc, sim))
        
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def stem_query(self, query):
        if self.corpus.stemmer:
            words = [self.corpus.stemmer.stem(w) for w in query.words]
        else:
            words = query.words
        return words
    
    def weight_query(self, ti: str, query_vect: nltk.Counter):
        freq = query_vect[ti]
        max_freq = query_vect.most_common(1)[0][1]
        tf = freq / max_freq
        idf = self.idf(ti)
        return (self.smoothing + (1 - self.smoothing) * tf) * idf
    
    def weight_doc(self, ti: str, dj: Document) -> float:
        return self.tf(ti, dj) * self.idf(ti)
    
    def tf(self, ti: str, dj: Document) -> float:
        freq = self.corpus.get_frequency(ti, dj)
        max_freq_tok, max_freq = self.corpus.get_max_frequency(dj)
        return freq / max_freq
    
    def idf(self, ti: str) -> float:
        N = len(self.corpus.documents)
        ni = self.corpus.vocabulary[ti]
        return math.log10(N / ni)
    
    def load_query(self, query):
        corpus_fb_path = self.feedback_path / self.corpus.name
        try:
            return json.load(open(corpus_fb_path / 'feedback.json')).get(query, None)
        except FileNotFoundError:
            return None
    
    def rocchio(self, query, dr, dnr, alpha=1, beta=0.75, gamma=0.15):
        if not self.processed_queries.get(query):
            return
        
        it_sum = lambda *x: map(sum, zip(*x))
        
        query_counter = self.processed_queries[query]
        w_q = [0 if word not in query_counter else self.weight_query(word, query_counter)
               for word in self.corpus.vocabulary.keys()]
        dr_q = [self.weight_doc_vector(id) for id in self.corpus.documents.keys() if id in dr]
        dnr_q = [self.weight_doc_vector(id) for id in self.corpus.documents.keys() if id in dnr]
        
        alpha_q = map(lambda x: x * alpha, w_q)
        sum_dr = it_sum(*dr_q)
        sum_dnr = it_sum(*dnr_q)
        
        sum_dr = map(lambda x: x * beta / len(dr), sum_dr)
        sum_dnr = map(lambda x: x * gamma / len(dnr), sum_dnr)
        
        return list(it_sum(alpha_q, sum_dr, sum_dnr))
    
    def weight_doc_vector(self, id):
        return [self.weight_doc(word, id) for word in self.corpus.vocabulary.keys()]
    
    def save_query(self, query, w_q):
        corpus_fb_path = self.feedback_path / self.corpus.name
        os.makedirs(corpus_fb_path, exist_ok=True)
        corpus_fb_path /= 'feedback.json'
        if not os.path.exists(corpus_fb_path):
            open(corpus_fb_path, 'x').close()
        with open(corpus_fb_path, 'r') as f:
            dic = json.load(f)
        dic[query] = w_q
        with open(corpus_fb_path, 'w') as f:
            json.dump(dic, f)
