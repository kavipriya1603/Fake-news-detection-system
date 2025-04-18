import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# Load the dataset (make sure 'news.csv' is in the same folder)
df = pd.read_csv("news.csv")
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})  # Encode labels

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train_tfidf, y_train)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and Vectorizer saved as model.pkl and vectorizer.pkl")
