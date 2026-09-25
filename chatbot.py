
import os
import pandas as pd
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KILOCODE_URL = "https://api.kilo.ai/api/gateway/v1/chat/completions"
KILOCODE_API_KEY = os.getenv("KILOCODE_API_KEY")

df = pd.read_excel("/content/icthub_dataset.xlsx")

responses = df["Chatbot Response"].dropna().astype(str).tolist()

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

response_vectors = vectorizer.fit_transform(responses)

def search_dataset(question, top_k=3):
    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        response_vectors
    )[0]

    best_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in best_indices:
        if similarities[index] > 0:
            results.append(responses[index])

    return results


def ask_chatbot(question):
    relevant_results = search_dataset(question, top_k=3)

    if relevant_results:
        context = "\n\n".join(relevant_results)
    else:
        context = "No relevant information was found."

    prompt = f"""
You are the ICTHub AI Assistant.

Use ONLY the information below.

ICTHub Information:
{context}

Question:
{question}

Answer:
"""

    resp = requests.post(
        KILOCODE_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KILOCODE_API_KEY}"
        },
        json={
            "model": "kilo-auto/free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        },
        timeout=90
    )

    data = resp.json()

    if resp.status_code == 200:
        return data["choices"][0]["message"]["content"]

    return f"Error {resp.status_code}: {data}"
