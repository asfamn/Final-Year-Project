import os
import sys
import logging

import numpy as np
import pandas as pd

import pickle
import config 
import unidecode, ast
import validators

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

import config
from ingredient_parser import ingredient_parser


def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.parsed.values:
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted



def get_recommendations(N, scores):
    df_recipes = pd.read_csv(config.PARSED_PATH)
    
    # Print columns for debugging
    # print("Columns in df_recipes:", df_recipes.columns)
    # print(df_recipes.head())

    
    # Order the scores and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]

    recommendation = pd.DataFrame(columns=["recipe", "ingredients", "score", "url", "instructions"])
    count = 0
    
    for i in top:
        # Print available columns for debugging
        # print("Available columns in df_recipes:", df_recipes.columns)
        
        # Check if 'instructions' exists in the DataFrame
        if 'instructions' in df_recipes.columns:
            recommendation.at[count, "instructions"] = df_recipes["instructions"][i]
        else:
            print("Error: 'instructions' not found in df_recipes columns.")
        
        # Continue with other columns
        recommendation.at[count, "url"] = df_recipes["recipe_urls"][i]
        recommendation.at[count, "recipe"] = ingredient_parser_final(df_recipes["recipe_name"][i])
        recommendation.at[count, "ingredients"] = title_parser(df_recipes["ingredients"][i])
        recommendation.at[count, "score"] = f"{scores[i]}"
        
        # # Validate and handle image URL
        # image_url = df_recipes["image_links"][i]
        # if validators.url(image_url):  # Check if the URL is valid
        #     recommendation.at[count, "image_links"] = image_url
        # else:
        #     recommendation.at[count, "image_links"] = None  # or handle accordingly
        
        count += 1

    return recommendation


def ingredient_parser_final(ingredient):
    # Escape single-quotes within the string
    ingredient = ingredient.replace("'", r"\'")
    
    # Wrap the ingredient in double-quotes to form a valid string
    ingredient = '"' + ingredient + '"'

    ingredients = ast.literal_eval('[' + ingredient + ']')
    ingredients = ",".join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

def title_parser(title):
    title = unidecode.unidecode(title)
    return title


# def ingredient_parser_final(ingredient):
#     """
#     neaten the ingredients being outputted
#     """
#     if isinstance(ingredient, list):
#         ingredients = ingredient
#     else:
#         ingredients = ast.literal_eval(ingredient)

#     ingredients = ",".join(ingredients)
#     ingredients = unidecode.unidecode(ingredients)
#     return ingredients


class MeanEmbeddingVectorizer(object):
    def __init__(self, word_model):
        self.word_model = word_model
        self.vector_size = word_model.wv.vector_size

    def fit(self):  # comply with scikit-learn transformer requirement
        return self

    def transform(self, docs):  # comply with scikit-learn transformer requirement
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector

    def word_average(self, sent):

        mean = []
        for word in sent:
            if word in self.word_model.wv.index_to_key:
                mean.append(self.word_model.wv.get_vector(word))

        if not mean:  # empty words
            # If a text is empty, return a vector of zeros.
            # logging.warning(
            #     "cannot compute average owing to no vector for {}".format(sent)
            # )
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def word_average_list(self, docs):

        return np.vstack([self.word_average(sent) for sent in docs])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word_model):

        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size

    def fit(self, docs):  


        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))

        tfidf = TfidfVectorizer()
        tfidf.fit(text_docs)


        max_idf = max(tfidf.idf_)
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        return self

    def transform(self, docs):

        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector

    def word_average(self, sent):

        mean = []
        for word in sent:
            if word in self.word_model.wv.index_to_key:
                mean.append(
                    self.word_model.wv.get_vector(word) * self.word_idf_weight[word]
                )  # idf weighted

        if not mean:  # empty words
            # If a text is empty, return a vector of zeros.
            # logging.warning(
            #     "cannot compute average owing to no vector for {}".format(sent)
            # )
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def word_average_list(self, docs):

        return np.vstack([self.word_average(sent) for sent in docs])


def get_recs(ingredients, N=10, mean=False):
    # Load your word2vec model
    model = Word2Vec.load("model/model_cbow.bin")
    model.init_sims(replace=True)
    if model:
        print("Successfully loaded model")

    data = pd.read_csv("input/df_parsed.csv")
    data["parsed"] = data.ingredients.apply(ingredient_parser)
    corpus = get_and_sort_corpus(data)

    if mean:
        # print("Using Mean Embeddings")  # Add this line for debugging
        # Get average embeddings for each document
        mean_vec_tr = MeanEmbeddingVectorizer(model)
        doc_vec = mean_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)
    else:
        # print("Using TF-IDF Weighted Embeddings")  # Add this line for debugging
        # Use TF-IDF as weights for each word embedding
        tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
        tfidf_vec_tr.fit(corpus)
        doc_vec = tfidf_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)

    # Create embedding for input text
    input_text = ingredients  # Renamed to avoid variable shadowing
    # Create tokens with elements
    input_text = input_text.split(",")
    # Parse ingredient list using the local ingredient_parser function
    input_text = ingredient_parser(input_text)
    # Get embeddings for the ingredient doc
    if mean:
        input_embedding = mean_vec_tr.transform([input_text])[0].reshape(1, -1)
    else:
        input_embedding = tfidf_vec_tr.transform([input_text])[0].reshape(1, -1)

    # Get cosine similarity between input embedding and all document embeddings
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)
    for i, score in enumerate(scores):
        print(f"Recipe: {data['recipe_name'][i]}| Score: {score}")
    recommendations = get_recommendations(N, scores)
    return recommendations

def RecSys(ingredients, N=5):


    # load in tdidf model and encodings 
    with open(config.TFIDF_ENCODING_PATH, 'rb') as f:
        tfidf_encodings = pickle.load(f)

    with open(config.TFIDF_MODEL_PATH, "rb") as f:
        tfidf = pickle.load(f)

    # parse the ingredients using my ingredient_parser 
    try: 
        ingredients_parsed = ingredient_parser(ingredients)
    except:
        ingredients_parsed = ingredient_parser([ingredients])
    
    # use our pretrained tfidf model to encode our input ingredients
    ingredients_tfidf = tfidf.transform([ingredients_parsed])

    # calculate cosine similarity between actual recipe ingreds and test ingreds
    cos_sim = map(lambda x: cosine_similarity(ingredients_tfidf, x), tfidf_encodings)
    scores = list(cos_sim)

    # Filter top N recommendations 
    recommendations = get_recommendations(N, scores)
    return recommendations

if __name__ == "__main__":
    input = "chicken thigh,tofu,onion"
    rec = get_recs(input)
    print(rec)

