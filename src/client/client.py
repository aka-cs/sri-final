from pathlib import Path

from src.client.utils import cache
from src.corpus.cran_corpus import CranCorpus
from src.corpus.test_corpus import TestCorpus
from src.mri.vector_mri import VectorMRI


def available_corpus():
    return \
        {
            "cran": {
                "name": "Cranfield",
                "corpus": CranCorpus
            },
            "test": {
                "name": "Test",
                "corpus": TestCorpus
            }
        }


@cache
def get_mri(corpus_key, stemming=False):
    corpus = available_corpus()[corpus_key]['corpus'](Path(f'../resources/corpus/{corpus_key}').resolve(), stemming=stemming)
    mri = VectorMRI(corpus, feedback_path=Path('../resources/feedback').resolve())
    return mri
