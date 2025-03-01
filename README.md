# Semantic Book Recommender
Project inspired by "LLM Course – Build a Semantic Book Recommender (Python, OpenAI, LangChain, Gradio)" : https://www.youtube.com/watch?v=Q7mS1VHm3Yw  

It aims to build an intelligent book recommendation system using Large Language Models (LLMs), vector search, and Gradio.

## 📌 Project Overview

The project consists of five main components:

- Text Data Cleaning: Preparing and cleaning book descriptions for analysis. (See [cleaning_data.py](cleaning_data.py))

- Vector Search & Database: Building a vector database for semantic search, enabling users to find books based on natural language queries. (See [vec_search.py](vec_search.py))

- Text Classification (Zero-Shot LLMs): Categorizing books into "fiction" or "non-fiction" using zero-shot learning. (See [zeroshot_classification.py](zeroshot_classification.py))

- Sentiment & Emotion Analysis: Extracting emotions from book descriptions to sort books by tone (e.g., suspenseful, joyful, sad). (See [sentiment-analysis.ipynb](sentiment_analysis.py))

- Gradio Web Application: Building an interactive UI to allow users to get book recommendations. (See [dashboard_gradio.py](dashboard_gradio.py))

## 📜 Credits

This project is based on the freeCodeCamp course: Build a Semantic Book Recommender with LLMs – Full Course. 

## 🛠 Changes & Improvements

- **Intermediate results** are stored to facilitate later processing and avoid retraining in case of modifications to downstream files.

- **Custom .env file required**: You must create your own .env file containing your OpenAI and Hugging Face API keys.

- **Alternative Hugging Face model used**: To avoid relying on OpenAI's paid model, a substitute model from Hugging Face is used. While it may not match OpenAI's accuracy, this project focuses on learning about LLMs rather than achieving the highest accuracy.

- **Variable name adjustments**: Some variable names have been modified for better readability while maintaining the original tutorial's logic.