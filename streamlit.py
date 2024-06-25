import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import validators


import config,reci_rec

def make_clickable(name, link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = name
    return f'<a target="_blank" href="{link}">{text}</a>'

def main():
    # # image = Image.open("input/wordcloud.png").resize((680, 150))
    # st.image(image)
    st.markdown("# *:cooking: Cookit! :cooking:*")

    st.text("")

    # Initialize session state
    if "execute_recsys" not in st.session_state:
        st.session_state.execute_recsys = False
        st.session_state.recipe_df_clean = None
        st.session_state.recipe_display = None
        st.session_state.recipes = None
        st.session_state.model_computed = False

    ingredients = st.text_input("Enter ingredients you would like to cook with(Enter at least 3 ingredients)")
    st.session_state.execute_recsys = st.button("Recommend me!")

    if st.session_state.execute_recsys:
        recipe = reci_rec.get_recs(ingredients)
        # link is the column with hyperlinks
        recipe["url"] = recipe.apply(
            lambda row: make_clickable(row["recipe"], row["url"]), axis=1
        )
        recipe_display = recipe[["recipe"]]

        st.session_state.recipe_df_clean = recipe.copy()
        st.session_state.recipe_display = recipe_display.to_html(escape=False)
        st.session_state.recipes = recipe.recipe.values.tolist()
        st.session_state.model_computed = True
        st.session_state.execute_recsys = False

    if st.session_state.model_computed:
        recipe_all_box = st.selectbox(
            "These are top 5 suggestions,select a dish that appeals to you.",
            ["Recommended recipe", "Recipe Detail"],
        )
        if recipe_all_box == "Recommended recipe":
            st.write(st.session_state.recipe_display, unsafe_allow_html=True)
        else:
            selection = st.selectbox(
                "Select a delicious recipe", options=st.session_state.recipes
            )
            selection_details = st.session_state.recipe_df_clean.loc[
                st.session_state.recipe_df_clean.recipe == selection
            ]

            st.write(f"Recipe: {selection_details.recipe.values[0]}")
            # st.markdown(f"Ingredients: {selection_details.ingredients.values[0]}")
            # st.markdown(f"Instructions: {selection_details.instructions.values[0]}", unsafe_allow_html=True)
            
            ingredients_list = [ingredient.strip() for ingredient in selection_details.ingredients.values[0].split(',')]
            ingredients_bullet_points = '\n'.join([f"- {ingredient}" for ingredient in ingredients_list])
            st.text(f"Ingredients:\n{ingredients_bullet_points}")

            instructions_list = [instruction.strip() for instruction in selection_details.instructions.values[0].split(',')]
            instructions_bullet_points = '\n'.join([f"- {instruction}" for instruction in instructions_list])
            st.text(f"Instructions:\n{instructions_bullet_points}")

            st.markdown(f"Learn more: {selection_details.url.values[0]}", unsafe_allow_html=True)
            st.write(f"Score: {selection_details.score.values[0]}")

            # image_links = selection_details.image_links.values[0]

            # if isinstance(image_links, list):
            #     image_url = image_links[0]  # Take the first URL from the list
            # else:
            #     image_url = image_links

            # try:
            #     if validators.url(image_url):  # Check if the URL is valid
            #         response = requests.get(image_url)
            #         response.raise_for_status()  # Check for HTTP errors
            #         img = Image.open(BytesIO(response.content))
            #         st.image(img, caption='Recipe Image', use_column_width=True)
            #     else:
            #         st.error(f"Error loading image: {e}")
            # except Exception as e:
            #     st.error(f"Invalid image URL: {image_url}")



if __name__ == "__main__":
    main()
