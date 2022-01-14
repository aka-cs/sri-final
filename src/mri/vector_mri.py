import json
import math
import os
from functools import reduce
from pathlib import Path

import nltk
from typing import List, Tuple

from src.corpus.corpus import Corpus, Document
from src.corpus.query import Query
from src.mri.mri import MRI


class VectorMRI(MRI):
    
    def __init__(self, corpus: Corpus, feedback_path=None, smoothing: int = 0.4):
        super().__init__(corpus, feedback_path)
        self.smoothing = smoothing
        self.processed_queries = {}
        # self.process_corpus()
    
    def query(self, query: Query) -> List[Tuple[Document, float]]:
        
        words = self.stem_query(query)
        words = [word for word in words if word in self.corpus.vocabulary]
        query_counter = nltk.Counter(words)
        ranking = []
        
        q_vector = None
        if self.feedback_path:
            q_vector = self.load_query(query.text)
        if q_vector is None:
            q_vector = {word: self.weight_query(word, query_counter) for word in query_counter.keys()}
            self.save_query(query.text, q_vector)
        
        for doc, counter in self.corpus.documents.items():
            
            num = 0
            doc_weight = self.weight_doc_dict(doc)
            doc_weights_sqr = sum([x**2 for x in doc_weight.values()])
            query_weights_sqr = 0
            for word, q_vector_i in q_vector.items():
                w_doc = doc_weight.get(word, 0)
                w_query = q_vector_i
                num += w_doc * w_query
                query_weights_sqr += w_query ** 2
            
            try:
                sim = num / (math.sqrt(doc_weights_sqr) * math.sqrt(query_weights_sqr))
            except ZeroDivisionError:
                sim = 0
            
            if sim > 0.1:
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
        return math.log(N / ni)
    
    def load_query(self, query):
        corpus_fb_path = self.feedback_path / self.corpus.name
        try:
            return json.load(open(corpus_fb_path / 'feedback.json')).get(query, None)
        except FileNotFoundError:
            return None
    
    def rocchio(self, query, dr, dnr, alpha=1, beta=0.75, gamma=0.15):
        
        def it_sum(*x):
            return map(sum, zip(*x))
        
        w_d = self.load_query(query)
        w_q = [w_d.get(word, 0) for word in self.corpus.vocabulary.keys()]
        dr_q = [self.weight_doc_vector(id) for id in self.corpus.documents.keys() if id in dr]
        dnr_q = [self.weight_doc_vector(id) for id in self.corpus.documents.keys() if id in dnr]
        
        alpha_q = map(lambda x: x * alpha, w_q)
        sum_dr = it_sum(*dr_q)
        sum_dnr = it_sum(*dnr_q)
        
        sum_dr = map(lambda x: x * beta / len(dr), sum_dr)
        sum_dnr = map(lambda x: x * gamma / len(dnr), sum_dnr)
        
        return {word: value for word, value in
                zip(self.corpus.vocabulary.keys(), it_sum(alpha_q, sum_dr, sum_dnr))
                if value > 0.3}

    def weight_doc_dict(self, doc):
        return {word: self.weight_doc(word, doc)
                for word in self.corpus.documents[doc].keys()}
    
    def weight_doc_vector(self, doc):
        return [0 if word not in self.corpus.documents[doc] else self.weight_doc(word, doc)
                for word in self.corpus.vocabulary.keys()]
    
    def save_query(self, query, w_q):
        corpus_fb_path = self.feedback_path / self.corpus.name
        os.makedirs(corpus_fb_path, exist_ok=True)
        corpus_fb_path /= 'feedback.json'
        if not os.path.exists(corpus_fb_path):
            open(corpus_fb_path, 'x').close()
        try:
            with open(corpus_fb_path, 'r') as f:
                dic = json.load(f)
        except json.decoder.JSONDecodeError:
            dic = {}
        dic[query] = w_q
        with open(corpus_fb_path, 'w') as f:
            json.dump(dic, f)

    def load_tfidf(self):
        try:
            indexed_corpus_path = Path(f'../resources/indexed_corpus/{self.corpus.name}/')
            with open(indexed_corpus_path / 'tfidf.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return
        except Exception as e:
            raise e
        
    def feedback(self, query, docs_id):
        dr = [self.corpus.id2doc[doc_id] for doc_id, rel in docs_id.items() if rel]
        dnr = [self.corpus.id2doc[doc_id] for doc_id, rel in docs_id.items() if not rel]
        nw = self.rocchio(query, dr, dnr)
        self.save_query(query, nw)
        return nw

    def save_tfidf(self, tfidf):
        indexed_corpus_path = Path(f'../resources/indexed_corpus/{self.corpus.name}/')
        os.makedirs(indexed_corpus_path, exist_ok=True)
        with open(indexed_corpus_path / 'tfidf.json', 'w') as f:
            json.dump(tfidf, f)
