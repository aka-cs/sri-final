import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from itertools import product
from functools import reduce
from string import punctuation


def tokenize(text: str, language: str = 'english', stemmer: nltk.SnowballStemmer = None):
    
    text = text.lower()
    
    words = nltk.word_tokenize(text)
    
    table = str.maketrans('', '', punctuation)
    tokens = map(lambda word: word.translate(table), words)
    
    tokens = list(filter(lambda x: x.isalpha(), tokens))
    
    stop_words = set(stopwords.words(language))
    tokens = filter(lambda x: x not in stop_words, tokens)
    
    tokens = filter(lambda x: len(x) > 1, tokens)
    
    if stemmer:
        tokens = map(stemmer.stem, tokens)
    
    return list(tokens)
    
