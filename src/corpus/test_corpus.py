from collections import Counter
from pathlib import Path

from src.corpus.corpus import Corpus


class TestCorpus(Corpus):
    
    def __init__(self, path: Path):
        super().__init__(path)
        self.process_corpus()

    def process_corpus(self):
        for doc in self.path.glob('*.txt'):
            with open(doc, 'r') as f:
                words = f.read().split()
            
            for w in set(words):
                self.vocabulary[w] = self.vocabulary.get(w, 0) + 1
            
            self.documents[str(doc.resolve().as_posix())] = Counter(words)
