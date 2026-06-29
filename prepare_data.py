import pandas as pd

def load_recipes(path="./RAW_recipes.csv"):

    df = pd.read_csv(path)

    print("Dataset Loaded: ", df.shape)

    return df

def build_doc(df):
    documents = []

    for _, row in df.iterrows():
        name = row.get("name", "")
        ingredients = row.get("ingredients", "")
        steps = row.get("steps: ", "")
        tags = row.get("tags", "")
        time = row.get("minutes", "")

        doc = f"""
           Recipe name: {name}

Ingredients: {ingredients}

Instrcutions: {steps}

Tags: {tags}

Cooking Time: {time} minutes
        """

        documents.append(doc)

    return documents    
