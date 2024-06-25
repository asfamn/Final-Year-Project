import requests
from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np
import os

headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"}

url = "https://www.jamieoliver.com/recipes/category/course/mains/"
page = requests.get(url)

class recipeDetail():

    def __init__(self, url):
        self.url = url 
        self.soup = BeautifulSoup(requests.get(url, headers=headers).content, 'html.parser')
    
    def recipe_name(self):
        try:
            return self.soup.find('h1').text.strip()
        except: 
            return np.nan
        
    def serves(self):
        try:
            return self.soup.find('div', {'class': 'recipe-detail serves'}).text.split(' ',1)[1]
        except:
            return np.nan 

    def cooking_time(self):
        try:
            return self.soup.find('div', {'class': 'recipe-detail time'}).text.split('In')[1]
        except:
            return np.nan


    def difficulty(self):
        try:
            return self.soup.find('div', {'class': 'col-md-12 recipe-details-col remove-left-col-padding-md'}).text.split('Difficulty')[1]
        except:
            return np.nan

    def ingredients(self):
        try:
            ingredients = [] 
            for li in self.soup.select('.ingred-list li'):
                ingred = ' '.join(li.text.split())
                ingredients.append(ingred)
            return ingredients
        except:
            return np.nan
    
    def instructions(self):
        try:
            instructions_wrapper = self.soup.find('div', class_='instructions-wrapper')
            if not instructions_wrapper:
                return ["Instructions not found"]

            method_p = instructions_wrapper.find('div', class_='method-p')
            if not method_p:
                return ["Method not found"]

            instructions = []
            for step in method_p.find_all('li'):
                text = step.text.strip()
                if text:
                    instructions.append(text)

            if not instructions:
                return ["No instructions found"]
            return instructions

        except Exception:
            return ["Unable to parse instructions"]


    def image_links(self):
        try:
            image_tags = self.soup.find_all('img')
            image_links = [img['src'] for img in image_tags if 'src' in img.attrs and img['src'].lower().endswith('.jpg?tr=w-800,h-1066')]
            return image_links
        except:
            return np.nan