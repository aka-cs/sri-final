from abc import abstractmethod, ABC
from pathlib import Path
from typing import List

from src.corpus.corpus import Corpus
from src.corpus.query import Query


class MRI(ABC):
    
    def __init__(self, corpus: Corpus, feedback_path=None):
        self.corpus: Corpus = corpus
        self.feedback_path = feedback_path
    
    @abstractmethod
    def query(self, query: Query) -> List[Path]:
        pass
