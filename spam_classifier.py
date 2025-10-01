# Email Spam Classifier Prototype
# Run this in Visual Studio / VS Code

# ------------------------------
# 1. Import libraries
# ------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import nltk
import string
import re

# Uncomment if you run first time
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.corpus import stopwords

# ------------------------------
# 2. Load dataset (UCI SMS Spam Dataset)
# ------------------------------
# Download dataset from Kaggle/UCI and save as "spam.csv" in same folder
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]

print("Dataset Sample:")
print(df.head())

# ------------------------------
# 3. Data Preprocessing
# ------------------------------
def clean_text(text):
    text = text.lower()                                # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)         # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)            # remove numbers/punctuations
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = nltk.word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

df["cleaned"] = df["message"].apply(clean_text)

# ------------------------------
# 4. Train-Test Split
# ------------------------------
X = df["cleaned"]
y = df["label"].map({"ham": 0, "spam": 1})  # ham=0, spam=1

# Convert text to numerical features
vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ------------------------------
# 5. Train Models
# ------------------------------

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Support Vector Machine
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# ------------------------------
# 6. Evaluation
# ------------------------------
print("\n=== Naive Bayes ===")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print("\n=== SVM ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# ------------------------------
# 7. Visualization of Keywords
# ------------------------------
spam_words = " ".join(df[df.label=="spam"]["cleaned"])
ham_words = " ".join(df[df.label=="ham"]["cleaned"])

spam_wc = pd.Series(spam_words.split()).value_counts().head(20)
ham_wc = pd.Series(ham_words.split()).value_counts().head(20)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.barplot(x=spam_wc.values, y=spam_wc.index, color="red")
plt.title("Top Spam Words")

plt.subplot(1,2,2)
sns.barplot(x=ham_wc.values, y=ham_wc.index, color="blue")
plt.title("Top Ham Words")

plt.show()
