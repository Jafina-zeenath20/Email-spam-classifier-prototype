# train_model.py
# Train model & save with Pickle

import pandas as pd
import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)     # remove numbers/punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["cleaned"] = df["message"].apply(clean_text)

X = df["cleaned"]
y = df["label"].map({"ham": 0, "spam": 1})

# Vectorize
vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_vec, y)

# Save model & vectorizer
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
