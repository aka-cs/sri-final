import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Counter, Tuple

import nltk


class Document:
    
    def __init__(self, id, title, author, text):
        self.id = id
        self.title = title
        self.text = text
        self.author = author
        
    def __eq__(self, other):
        return isinstance(other, Document) and self.id == other.id
    
    def __hash__(self):
        return hash(self.id)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'text': self.text,
            'author': self.author
        }
    
    def __str__(self):
        return str(self.id)
    
    def __repr__(self):
        return repr(self.to_dict())


class Corpus(ABC):
    
    def __init__(self, path: Path, name: str = 'corpus', language: str = 'english', stemming=False):
        self.name = name
        self.language = language
        self.stemmer = nltk.SnowballStemmer(language) if stemming else None
        self.path = path
        
        try:
            self.load_indexed_corpus()
        except FileNotFoundError:
            self.documents: Dict[Document, Counter[str, int]] = {}
            self.vocabulary: Dict[str, int] = {}
            self.process_corpus()
            self.save_indexed_corpus()
        
    @abstractmethod
    def process_corpus(self):
        raise NotImplementedError

    def load_indexed_corpus(self):
        indexed_corpus_path = Path(f'../resources/indexed_corpus/{self.name}/')
        stemmed = '' if not self.stemmer else '_stemmed'
        self.documents = pickle.load(open(indexed_corpus_path / f'docs{stemmed}.pkl', 'rb'))
        self.vocabulary = pickle.load(open(indexed_corpus_path / f'vocab{stemmed}.pkl', 'rb'))

    def save_indexed_corpus(self):
        indexed_corpus_path = Path(f'../resources/indexed_corpus/{self.name}/')
        os.makedirs(indexed_corpus_path, exist_ok=True)
        stemmed = '' if not self.stemmer else '_stemmed'
        pickle.dump(self.documents, open(indexed_corpus_path / f'docs{stemmed}.pkl', 'wb'))
        pickle.dump(self.vocabulary, open(indexed_corpus_path / f'vocab{stemmed}.pkl', 'wb'))

    def get_frequency(self, tok_id: str, doc_id: Document) -> int:
        vector = self.documents[doc_id]
        try:
            return vector[tok_id]
        except KeyError:
            return 0

    def get_max_frequency(self, doc: Document) -> Tuple[str, int]:
        vector = self.documents[doc]
        return vector.most_common(1)[0]
