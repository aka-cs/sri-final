from collections import Counter
from pathlib import Path

from src.corpus.corpus import Corpus
from src.corpus.utils import tokenize


class TestCorpus(Corpus):
    
    def __init__(self, path: Path, stemming=False):
        super().__init__(path, 'test', 'english', stemming)

    def process_corpus(self):
        for doc in self.path.glob('*.txt'):
            with open(doc, 'r') as f:
                words = tokenize(f.read(), self.language, self.stemmer)
            
            for w in set(words):
                self.vocabulary[w] = self.vocabulary.get(w, 0) + 1
            
            self.documents[str(doc.resolve().as_posix())] = Counter(words)
