import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("dataset.csv")

print("Columns:", data.columns)

# Keep only valid labels
data = data[data["label"].isin(["Accepted", "Rejected"])]

# Remove nulls PROPERLY
data = data.dropna(subset=["judgement", "label"])

# Convert to string (VERY IMPORTANT)
data["judgement"] = data["judgement"].astype(str)

# 🔥 LIMIT DATA (VERY IMPORTANT FOR SPEED)
data = data.sample(n=10000, random_state=42)  # use only 5k rows

# Shorten long texts (VERY IMPORTANT)
data["judgement"] = data["judgement"].apply(lambda x: x[:1000])

# Input & Output
X = data["judgement"]
y = data["label"]

# Convert labels
y = y.map({
    "Accepted": 1,
    "Rejected": 0
})

# Vectorization (LIMIT FEATURES)
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Faster model
model = LogisticRegression(max_iter=100, solver='liblinear')
model.fit(X_vec, y)

# Save
os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("✅ Model trained successfully!")