from abc import ABC
from pathlib import Path
from typing import Dict, Counter


class Corpus(ABC):
    
    def __init__(self, path: Path):
        self.path = path
        self.documents: Dict[str, Counter[str, int]] = {}
        self.vocabulary: Dict[str, int] = {}
