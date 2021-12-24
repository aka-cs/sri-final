from pathlib import Path

from src.corpus.query import Query
from src.corpus.test_corpus import TestCorpus
from src.mri.vector_mri import VectorMRI

if __name__ == '__main__':
    corpus = TestCorpus(Path('../resources/corpus/test').absolute())
    
    mri = VectorMRI(corpus)
    
    q = Query('1663 1841 1842')
    
    print(mri.query(q))
