
import pandas as pd

# Load the existing DataFrames
recipe_urls_df = pd.read_csv("C:/Users/Kaizen/Desktop/cookit_Final/input/recipe_urls.csv")
df_recipes_df = pd.read_csv("C:/Users/Kaizen/Desktop/cookit_Final/input/df_recipes.csv")

# Concatenate the DataFrames
concatenated_df = pd.concat([recipe_urls_df, df_recipes_df], axis=1)

# Save the concatenated DataFrame to a new CSV file
concatenated_csv_path = "C:/Users/Kaizen/Desktop/cookit_Final/input/recipe_detail.csv"
concatenated_df.to_csv(concatenated_csv_path, index=False)

print("Concatenation complete!")
