from collections import Counter
from pathlib import Path

from src.corpus.corpus import Corpus, Document
from src.corpus.utils import tokenize


class TestCorpus(Corpus):
    
    def __init__(self, path: Path, stemming=False):
        super().__init__(path, 'test', 'english', stemming)

    def process_corpus(self):
        for doc_path in self.path.glob('*.txt'):
            
            with open(doc_path, 'r') as f:
                document = Document(doc_path.resolve().as_posix(),
                                    doc_path.name,
                                    doc_path.resolve().as_posix(),
                                    f.read())
            
            words = tokenize(document.text, self.language, self.stemmer)
            
            if not words:
                continue
            
            for w in set(words):
                self.vocabulary[w] = self.vocabulary.get(w, 0) + 1
            
            self.documents[document] = Counter(words)
