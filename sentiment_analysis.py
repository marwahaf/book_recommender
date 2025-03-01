import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

pyfile_path = os.path.abspath(__file__)
directory = os.path.dirname(pyfile_path)
cleandata_path = os.path.join(directory, "data_results/books_with_categories.csv")
books = pd.read_csv(cleandata_path)


classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
)
classifier("i love life!")

description = books["description"][0]

# Whole description classifier
classifier(description)

#  entence by sentence classifier
classifier(description.split("."))


emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
isbn = []
emotion_scores = {label: [] for label in emotion_labels}


def calculate_max_emotion_scores(predictions):
    per_emotion_scores = {label: [] for label in emotion_labels}
    for prediction in predictions:
        sorted_predictions = sorted(prediction, key=lambda x: x["label"])
        for index, label in enumerate(emotion_labels):
            per_emotion_scores[label].append(sorted_predictions[index]["score"])
    return {label: np.max(scores) for label, scores in per_emotion_scores.items()}


emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
isbn = []
emotion_scores = {label: [] for label in emotion_labels}

for i in tqdm(range(len(books))):
    isbn.append(books["isbn13"][i])
    sentences = books["description"][i].split(".")
    predictions = classifier(sentences)
    max_scores = calculate_max_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])

emotions_df = pd.DataFrame(emotion_scores)
emotions_df["isbn13"] = isbn
books = pd.merge(books, emotions_df, on="isbn13")

ouput_path = os.path.join(directory, "data_results/books_with_emotions.csv")
books.to_csv(ouput_path, index=False)
