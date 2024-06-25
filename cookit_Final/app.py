# app.py
import os
from flask import Flask, jsonify, request, render_template
import json, requests, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ingredient_parser import ingredient_parser
import config, reci_rec
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
model_path = os.path.join(os.path.dirname(__file__), config.TFIDF_MODEL_PATH)

# try:
#     nltk.data.find("corpora/wordnet")
# except LookupError:
#     nltk.download("wordnet")

app = Flask(__name__, template_folder='templates')
@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

def hello():
    # This is the homepage of our API.
    # It can be accessed by http://127.0.0.1:5000/
    return HELLO_HTML
HELLO_HTML = """
     <html><body>
         <h1>Welcome to my api: Whatscooking!</h1>
         <p>Please add some ingredients to the url to receive recipe recommendations.
            You can do this by appending "/recipe?ingredients= Pasta Tomato ..." to the current url.
         <br>Click <a href="/recipe?ingredients=pasta,tomato,onion">here</a> for an example when using the ingredients: pasta, tomato and onion.
     </body></html>
     """
@app.route('/recipe/detail/<recipe_name>', methods=["GET"])
def recipe_detail(recipe_name):
    # Retrieve the details of the selected recipe based on the recipe name
    # You may need to modify this part based on how your data is structured
    recipe_details = get_recipe_details(recipe_name)
    
    # Render the recipe detail template with the details
    return render_template('recipe_detail.html', recipe_details=recipe_details)

@app.route('/recipe', methods=["GET"])
def recommend_recipe():
    ingredients = request.args.get('ingredients')
    recipe = reci_rec.get_recs(ingredients)

    response = []
    for _, row in recipe.iterrows():
        recipe_data = {
            'recipe': str(row['recipe']),
            'ingredients': str(row['ingredients']),
            'score': str(row['score']),
            'url': str(row['url'])
        }
        response.append(recipe_data)

    return jsonify(response)




if __name__ == "__main__":
   print("Starting the application")  
   app.run(host="0.0.0.0", debug=True)
   print("Application terminated")
