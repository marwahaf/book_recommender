import kagglehub
import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("TkAgg")
import os

import matplotlib.pyplot as plt
import numpy as np

# Initializing
pyfile_path = os.path.abspath(__file__)
directory = os.path.dirname(pyfile_path)

# Download latest version
path = "dylanjcastillo/7k-books-with-metadata"
# Télécharger le dataset seulement si il n'existe pas
if not os.path.exists(path):
    print("Dataset not found. Downloading...")
    path = kagglehub.dataset_download(
        "imsparsh/multimodal-mirex-emotion-dataset", download_path="/app/datasets"
    )
else:
    print("Dataset already exists.")
books = pd.read_csv(f"{path}/books.csv")


column_list = books.columns.tolist()
print("column list ", column_list)

# Get basic statistics (mean, std, min, max, 25%, 50%, 75%)
print("Data statistics : ")
print(books.describe())

# Creating a heatmap to see the missing data's distribution
fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)
plt.xlabel("Columns")
plt.ylabel("Missing values")
# plt.show()

# correlation matrix creation
books["missing_description"] = np.where(books["description"].isna(), 1, 0)
books["age_of_book"] = 2024 - books["published_year"]
columns_of_interest = [
    "num_pages",
    "age_of_book",
    "missing_description",
    "average_rating",
]
correlation_matrix = books[columns_of_interest].corr(method="spearman")

# correclation matrix plot
sns.set_theme(style="white")
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"label": "Spearman correlation"},
)
heatmap.set_title("correlation heatmap")
# plt.show()

# number books with missing values
books_with_na = books[
    books["description"].isna()
    | books["num_pages"].isna()
    | books["age_of_book"].isna()
    | books["published_year"].isna()
]
print("Books missing data (descripton , num_pages , age of book , published year) : ")
print(books_with_na)

# Getting rid of na values
books_missing = books[
    ~books["description"].isna()
    & ~books["num_pages"].isna()
    & ~books["age_of_book"].isna()
    & ~books["published_year"].isna()
]
print("Books cleaned : ", books_missing)
print("cleaned data  statistics : ")
print(books_missing.describe())

# Ditribution per categories
categories_distribution = (
    books_missing["categories"]
    .value_counts()
    .reset_index()
    .sort_values("count", ascending=False)
)
print("categories_distribution", categories_distribution)

# Count numbers of words in the description
books_missing["words_in_description"] = (
    books_missing["description"].str.split().str.len()
)
plt.figure(figsize=(10, 6))
sns.histplot(
    x=books_missing["words_in_description"].values,
    y=books_missing["words_in_description"].count(),
)
plt.title("number of words in description")
# plt.show()

bad_decriptions = books_missing.loc[
    books_missing["words_in_description"].between(1, 4), "description"
]
medium_decriptions = books_missing.loc[
    books_missing["words_in_description"].between(5, 24), "description"
]
good_decriptions = books_missing.loc[
    books_missing["words_in_description"].between(25, 34), "description"
]

print("bad_decriptions", bad_decriptions)
print("medium_decriptions", medium_decriptions)
print("good_decriptions", good_decriptions)

books_missing_25_words = books_missing[books_missing["words_in_description"] >= 25]
print("books with more than 25_words in description : ", books_missing_25_words.shape)

books_missing_25_words["title_and_subtitle"] = np.where(
    books_missing_25_words["subtitle"].isna(),
    books_missing_25_words["title"],
    books_missing_25_words[["title", "subtitle"]].astype(str).agg(" ".join, axis=1),
)
print(books_missing_25_words["title_and_subtitle"])

books_missing_25_words["tagged_description"] = (
    books_missing_25_words[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)
)
print("existant columns ", books_missing_25_words.columns.tolist())

# Saving the cleaned csv
ouput_path = os.path.join(directory, "data_results/books_cleaned.csv")
books_missing_25_words.drop(
    ["subtitle", "missing_description", "age_of_book", "words_in_description"], axis=1
).to_csv(ouput_path, index=False)
