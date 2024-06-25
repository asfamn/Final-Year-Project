import os
import sys
import logging
import unidecode
import ast

import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

import config
from ingredient_parser import ingredient_parser

# get corpus with the documents sorted in alphabetical order
def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.parsed.values:
        if isinstance(doc, str):
            # Handle the case where 'parsed' is a string
            doc = doc.split()  # Split the string into a list of words
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted
  
# calculate average length of each document 
def get_window(corpus):
    lengths = [len(doc) for doc in corpus]
    avg_len = float(sum(lengths)) / len(lengths)
    return round(avg_len)

if __name__ == "__main__":
    # load in data
    data = pd.read_csv('C:/Users/Kaizen/Desktop/cookit_Final/input/df_parsed.csv')
    # parse the ingredients for each recipe
    data['parsed'] = data.ingredients.apply(ingredient_parser)
    # get corpus
    corpus = get_and_sort_corpus(data)
    print(f"Length of corpus: {len(corpus)}")
    # train and save CBOW Word2Vec model
    model_cbow = Word2Vec(
      corpus, sg=0, workers=8, window=get_window(corpus), min_count=1, vector_size=100
    )
    model_cbow.save('C:/Users/Kaizen/Desktop/cookit_Final/model/model_cbow.bin')
    print("Word2Vec model successfully trained")