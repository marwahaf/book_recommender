import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import pipeline

# api key retrieval
load_dotenv()

pyfile_path = os.path.abspath(__file__)
directory = os.path.dirname(pyfile_path)
cleandata_path = os.path.join(directory, "data_results/books_cleaned.csv")

books = pd.read_csv(cleandata_path)

categories_distribution = books["categories"].value_counts().reset_index()
print("categories_distribution", categories_distribution)

categories_filtered = (
    books["categories"].value_counts().reset_index().query("count>=50")
)
print("categories_distribution filtered", categories_filtered)

category_mapping = {
    "Fiction": "Fiction",
    "Juvenile Fiction": "Children's Fiction",
    "Biography & Autobiography": "Nonfiction",
    "History": "Nonfiction",
    "Literary Criticism": "Nonfiction",
    "Philosophy": "Nonfiction",
    "Religion": "Nonfiction",
    "Comics & Graphic Novels": "Fiction",
    "Drama": "Fiction",
    "Juvenile Nonfiction": "Children's Nonfiction",
    "Science": "Nonfiction",
    "Poetry": "Fiction",
}

books["simple_categories"] = books["categories"].map(category_mapping)
print(books[~books["simple_categories"].isna()])
print("number of books categorised : ", books["simple_categories"].count())

# bart large mnli facebook
fiction_categories = ["Fiction", "Nonfiction"]
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sequence = books.loc[
    books["simple_categories"] == "Fiction", "description"
].reset_index(drop=True)[0]
print(sequence)
print(pipe(sequence, fiction_categories))

max_index = np.argmax(pipe(sequence, fiction_categories)["scores"])
max_label = pipe(sequence, fiction_categories)["labels"][max_index]
print(" these books are most likely :", max_label)


def generate_prediction(sequence, categories):
    predictions = pipe(sequence, categories)
    max_index = np.argmax(predictions["scores"])
    max_label = pipe(sequence, fiction_categories)["labels"][max_index]
    return max_label


actual_cats = []
predicted_cats = []

# just to know how long we have left in the loop
for i in tqdm(range(0, 300)):
    sequence = books.loc[
        books["simple_categories"] == "Fiction", "description"
    ].reset_index(drop=True)[i]
    predicted_cats += [generate_prediction(sequence, fiction_categories)]
    actual_cats += ["Fiction"]

for i in tqdm(range(0, 300)):
    sequence = books.loc[
        books["simple_categories"] == "Nonfiction", "description"
    ].reset_index(drop=True)[i]
    predicted_cats += [generate_prediction(sequence, fiction_categories)]
    actual_cats += ["Nonfiction"]


predictions_df = pd.DataFrame(
    {"actual categories": actual_cats, "predicted categories": predicted_cats}
)
predictions_df["correc_prediction"] = np.where(
    predictions_df["actual categories"] == predictions_df["predicted categories"], 1, 0
)

percentage_correct = predictions_df["correc_prediction"].sum() / len(predictions_df)
print("percentage of correct predictions:", percentage_correct * 100)

isbns = []
predicted_cats = []

missing_cats = books.loc[
    books["simple_categories"].isna(), ["isbn13", "description"]
].reset_index(drop=True)

for i in tqdm(range(0, len(missing_cats))):
    sequence = missing_cats["description"][i]
    predicted_cats += [generate_prediction(sequence, fiction_categories)]
    isbns += [missing_cats["isbn13"][i]]

missing_predicted_df = pd.DataFrame(
    {"isbn13": isbns, "predicted_categories": predicted_cats}
)

# Merging prediction with original missing dataframe
books = pd.merge(books, missing_predicted_df, on="isbn13", how="left")
books["simple_categories"] = np.where(
    books["simple_categories"].isna(),
    books["predicted_categories"],
    books["simple_categories"],
)
books = books.drop(columns=["predicted_categories"])

print("specific fiction categories ")
print(
    books[
        books["categories"]
        .str.lower()
        .isin(
            [
                "romance",
                "science fiction",
                "scifi",
                "fantasy",
                "horror",
                "mystery",
                "thriller",
                "comedy",
                "crime",
                "historical",
            ]
        )
    ]
)

ouput_path = os.path.join(directory, "data_results/books_with_categories.csv")
books.to_csv(ouput_path, index=False)
