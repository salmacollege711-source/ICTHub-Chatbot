
import os
from pathlib import Path

import pandas as pd
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KILOCODE_URL = "https://api.kilo.ai/api/gateway/v1/chat/completions"
KILOCODE_API_KEY = os.getenv("KILOCODE_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "icthub_dataset.xlsx"

df = pd.read_excel(DATA_PATH)

responses = df["Chatbot Response"].dropna().astype(str).tolist()
