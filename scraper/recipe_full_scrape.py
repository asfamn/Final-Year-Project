import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import numpy as np
import os
from recipe_att import recipeDetail

attribs = ['recipe_name','ingredients','instructions','image_links']

results = pd.DataFrame(columns=attribs)

recipe_df = pd.read_csv("C:/Users/Kaizen/Desktop/cookit_Final/input/recipe_urls.csv")

# Scrape each recipe URL
for i, url in enumerate(recipe_df['recipe_urls']):

    # Print progress
    print(f"Scraping recipe {i+1}/{len(recipe_df['recipe_urls'])}")

    scraper = recipeDetail(url)
    data = [getattr(scraper, a)() for a in attribs]

    results.loc[i] = data

# Save the results to CSV
directory = "input"
if not os.path.exists(directory):
    os.makedirs(directory)

csv_path = "C:/Users/Kaizen/Desktop/cookit_Final/input/df_recipes.csv"
results.to_csv(csv_path, index=False)
print("Scraping complete!")
