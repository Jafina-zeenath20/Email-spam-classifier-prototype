📧 Email Spam Classifier using NLP & Machine Learning
📌 Overview

In the age of digital communication, spam emails pose serious threats by flooding inboxes with irrelevant or malicious content. This project implements a Machine Learning-based Spam Classifier that distinguishes between spam and legitimate (ham) emails using Natural Language Processing (NLP) techniques and supervised learning algorithms.

The system achieves 97–98% accuracy and includes a Tkinter GUI for real-time classification, plus visualization of spam-indicating keywords.

🚀 Features

✅ Detects whether an email is spam or ham

✅ Uses NLP preprocessing (tokenization, stopword removal, TF-IDF)

✅ Lightweight & fast, can integrate into email systems

✅ Real-time predictions with Tkinter GUI

✅ High accuracy with low false positives

✅ Option to visualize important spam keywords (Word Cloud + Bar Chart)

🛠️ Technologies Used

Language: Python

Libraries:

scikit-learn → Model training & evaluation

pandas, numpy → Data handling

NLTK → Text preprocessing

matplotlib, seaborn, wordcloud → Visualizations

tkinter → GUI

Algorithms Used: Naive Bayes, Logistic Regression, Support Vector Machine (SVM)

Dataset: UCI SMS Spam Collection Dataset

📂 Project Structure
Email-Spam-Classifier/
│
├── spam_classifier.py       # Model training & evaluation
├── spam_gui_visual.py       # Tkinter GUI with visualization
├── spam.csv                 # Dataset (downloaded separately)
├── models/
│   ├── spam_model.pkl       # Saved trained model
│   └── vectorizer.pkl       # Saved TF-IDF vectorizer
├── outputs/
│   ├── wordcloud.png        # Word Cloud of spam keywords
│   └── spam_bar_chart.png   # Bar chart of top spam words
└── README.md                # Project documentation

⚙️ Installation & Setup

Clone this repository:

git clone https://github.com/your-username/Email-Spam-Classifier.git
cd Email-Spam-Classifier


Install dependencies:

pip install -r requirements.txt


Example requirements.txt:

pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
wordcloud
tkinter


Download necessary NLTK resources:

import nltk
nltk.download('punkt')
nltk.download('stopwords')


Run the training script (to train and save model):

python spam_classifier.py


Launch the GUI application:

python spam_gui_visual.py

📊 Results

Naive Bayes → 97.7% accuracy

Logistic Regression → 94.7% accuracy

SVM → 97.8% accuracy

Confusion matrices and classification reports confirm low false positives and high precision.

🎨 Visualizations

Word Cloud of Spam Words

Top 20 Spam Words (Bar Chart)

These help analyze the most frequent spam keywords.

🐞 Troubleshooting

FileNotFoundError: 'spam.csv' → Ensure dataset is in the same folder as your script.

NLTK LookupError (punkt not found) → Run nltk.download('punkt').

GUI not opening → Make sure you’re running in a local environment (not Google Colab).

📌 Future Enhancements

Extend to phishing detection using URL/domain analysis

Deploy as a Flask/Django web app

Integrate with real email APIs (Gmail, Outlook)

👩‍💻 Author

Jafina Zeenath

🎓 BE CSE | Aspiring AI Engineer | Tech Enthusiast

💼 Passionate about ML, NLP, and real-world applications

🌐 GitHub: your-username

✨ If you like this project, don’t forget to ⭐ the repo!
