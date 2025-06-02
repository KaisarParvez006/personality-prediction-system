import docx2txt
import os
import re
import string
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Make sure these are downloaded
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load CV (replace with your actual file path)
cv_text = docx2txt.process("sample_cv.docx")

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

cleaned_text = preprocess(cv_text)

# Fake personality classifier (trained model placeholder)
# In a real project, you'd train this on a dataset like the Big Five Personality test
model = joblib.load("personality_model.pkl")  # You must have a trained model
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # TF-IDF vectorizer used during training

features = vectorizer.transform([cleaned_text])
prediction = model.predict(features)

print(f"Predicted Personality Trait: {prediction[0]}")