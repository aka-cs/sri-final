from pathlib import Path

from src.corpus.cran_corpus import CranCorpus, CranDocument
from src.corpus.query import Query
from src.corpus.test_corpus import TestCorpus
from src.mri.vector_mri import VectorMRI


def terminal_main():
    
    # corpus = CranCorpus(Path('../resources/corpus/cran').resolve(), stemming=True)
    corpus = TestCorpus(Path('../resources/corpus/test').resolve(), stemming=True)
    
    print('Corpus Built')
    
    # print(corpus.queries)
    
    mri = VectorMRI(corpus)
    #
    print('MRI built')
    #
    # q = Query('what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft')
    q = Query('democratic universe office native worldwide spell')
    #
    print('Processing Query')
    #
    r = mri.query(q)
    #
    # print(r)
    #
    d_r = {n: (i, rank) for i, (n, rank) in enumerate(r)}
    #
    print(d_r)
    
    return d_r


def web_main():
    from flask import Flask, request
    from flask_.flask_ import app
    import sys
    
    port = 5100
    
    if sys.argv.__len__() > 1:
        port = sys.argv[1]
        
    print("App running on port : {} ".format(port))
    
    app.run(host="localhost", port=port)


if __name__ == '__main__':
    
    flask = False
    
    if flask:
        web_main()
    else:
        terminal_main()
