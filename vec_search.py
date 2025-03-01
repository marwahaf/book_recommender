import os

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

#  api key retrieval
load_dotenv()

pyfile_path = os.path.abspath(__file__)
directory = os.path.dirname(pyfile_path)
cleandata_path = os.path.join(directory, "data_results/books_cleaned.csv")

books = pd.read_csv(cleandata_path)

# test loader langchain doesn"t work with pd dataframe
ouput_path = os.path.join(directory, "data_results/tagged_description.txt")
books["tagged_description"].to_csv(ouput_path, sep="\n", index=False, header=False)

raw_doc = TextLoader("data_results/tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_doc)

# Check if if we get only the descrciption
print(documents[0])

db_books = Chroma(documents, embedding_function=OpenAIEmbeddings())
query = "a book to teach children about nature"
docs = db_books.similarity_search(query, k=10)
query_result = books[
    books["isbn13"] == int(documents[0].page_content.split()[0].strip())
]
print(query_result)


def retrieve_semantic_recommandation(query, top_k) -> pd.DataFrame:
    """return top_k rows repondant a  la queery"""
    recs = db_books.similarity_search(query, k=50)
    books_list = []
    for i in range(0, len(recs)):
        books_list += [int(recs[i].page_content.strip('"').split()[0])]

    return books[books["isbn13"].isin(books_list).head(top_k)]
