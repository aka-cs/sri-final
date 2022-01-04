from dataclasses import dataclass
from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords, wordnet

from src.corpus.corpus import Corpus
from src.corpus.query import Query
from src.corpus.utils import tokenize


@dataclass
class CranDocument:
    id: str
    title: str
    author: str
    bibliography: str
    words: str


class CranQuery(Query):
    
    def __init__(self, id, text):
        super().__init__(text)
        self.id = id
        
    def __repr__(self):
        return f'{self.id}: {self.words}'
    
    def __str__(self):
        return repr(self)


class CranCorpus(Corpus):
    
    def __init__(self, path: Path, stemming=False):
        super(CranCorpus, self).__init__(path, 'cran', 'english', stemming)

    def process_corpus(self):
        
        doc_file = self.path.joinpath('cran.all.1400')
        
        with open(doc_file, 'r') as f:
            doc_text = f.read()
            
        reg = re.compile(r'.I +(?P<id>\d+) *\n'
                         r'.T *\n'
                         r'(?P<title>(?:.|\n)+?)'
                         r'.A *\n'
                         r'(?P<author>(?:.|\n)+?)'
                         r'.B *\n'
                         r'(?P<bibliography>(?:.|\n)+?)'
                         r'.W *\n'
                         r'(?P<words>(?:.|\n)+?)'
                         r'(?=.I)')
        
        docs = reg.findall(doc_text)
        docs = list(map(lambda x: CranDocument(*x), docs))
        for doc in docs:
            doc_text = '\n'.join([doc.title, doc.author, doc.bibliography, doc.words])
            words = tokenize(doc_text, self.language, self.stemmer)
            
            for w in set(words):
                self.vocabulary[w] = self.vocabulary.get(w, 0) + 1

            self.documents[doc.id] = nltk.Counter(words)