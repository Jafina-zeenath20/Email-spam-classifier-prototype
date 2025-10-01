# spam_gui_visual.py
# Email Spam Classifier GUI with Word Cloud + Bar Chart

import tkinter as tk
from tkinter import scrolledtext, messagebox, Toplevel
import pickle
import re
import string
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image, ImageTk

# ------------------------------
# Load trained model & vectorizer
# ------------------------------
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]

# ------------------------------
# Preprocessing function
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# ------------------------------
# Check spam function
# ------------------------------
def check_spam():
    email_text = text_area.get("1.0", tk.END).strip()
    if not email_text:
        messagebox.showwarning("Input Error", "Please enter email text.")
        return
    
    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    
    if pred == 1:
        result_label.config(text="🚨 SPAM Email", fg="red")
    else:
        result_label.config(text="✅ Legitimate Email (Ham)", fg="green")

# ------------------------------
# Show Word Cloud + Bar Chart
# ------------------------------
def show_visuals():
    spam_texts = df[df.label=="spam"]["message"].apply(clean_text)
    combined_text = " ".join(spam_texts)
    
    # ---------------- Word Cloud ----------------
    wc = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(combined_text)
    
    # ---------------- Top 20 words Bar Chart ----------------
    words = combined_text.split()
    word_counts = Counter(words)
    top_words = dict(word_counts.most_common(20))
    
    # Create new window
    vis_window = Toplevel(root)
    vis_window.title("Spam Keywords Visuals")
    vis_window.geometry("900x900")
    
    # Word Cloud Image
    plt.figure(figsize=(8,4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig("spam_wc.png")
    plt.close()
    
    wc_img = Image.open("spam_wc.png")
    wc_img = wc_img.resize((850,400))
    wc_img_tk = ImageTk.PhotoImage(wc_img)
    
    wc_label = tk.Label(vis_window, image=wc_img_tk)
    wc_label.image = wc_img_tk
    wc_label.pack(pady=5)
    
    # Bar Chart Image
    plt.figure(figsize=(8,4))
    plt.bar(top_words.keys(), top_words.values(), color="red")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Frequency")
    plt.title("Top 20 Spam Words")
    plt.tight_layout()
    plt.savefig("spam_bar.png")
    plt.close()
    
    bar_img = Image.open("spam_bar.png")
    bar_img = bar_img.resize((850,400))
    bar_img_tk = ImageTk.PhotoImage(bar_img)
    
    bar_label = tk.Label(vis_window, image=bar_img_tk)
    bar_label.image = bar_img_tk
    bar_label.pack(pady=5)

# ------------------------------
# Tkinter Layout
# ------------------------------
root = tk.Tk()
root.title("Email Spam Classifier")
root.geometry("650x550")
root.config(bg="#f4f4f4")

title = tk.Label(root, text="📧 Email Spam Classifier", font=("Arial", 18, "bold"), bg="#f4f4f4")
title.pack(pady=10)

text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=10, font=("Arial", 12))
text_area.pack(padx=10, pady=10)

check_button = tk.Button(root, text="Check Email", font=("Arial", 14), bg="#4CAF50", fg="white", command=check_spam)
check_button.pack(pady=5)

vis_button = tk.Button(root, text="Show Spam Keywords", font=("Arial", 14), bg="#FF5722", fg="white", command=show_visuals)
vis_button.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="#f4f4f4")
result_label.pack(pady=10)

root.mainloop()
