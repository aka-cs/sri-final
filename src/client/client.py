from pathlib import Path

from src.corpus.cran_corpus import CranCorpus
from src.corpus.test_corpus import TestCorpus
from src.mri.vector_mri import VectorMRI


def available_corpus():
    return \
        {
            "cran": {
                "name": "Cranfield",
                "class": CranCorpus
            },
            "test": {
                "name": "Test",
                "class": TestCorpus
            }
        }


def get_mri(corpus_key, stemming=False):
    corpus = available_corpus()[corpus_key]['class'](Path(f'../resources/corpus/{corpus_key}').resolve(), stemming=stemming)
    mri = VectorMRI(corpus)
    return mri
