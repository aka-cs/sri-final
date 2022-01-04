import nltk

from src.corpus.utils import tokenize


class Query:
    
    def __init__(self, text: str):
        self.text = text
        self.words = tokenize(self.text)
        
    def __iter__(self):
        for w in self.words:
            yield w
