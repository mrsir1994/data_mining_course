import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

def getData(categories,subset,shuffle,random_state):
    return fetch_20newsgroups(subset = subset, categories = categories, shuffle=shuffle, random_state = random_state)