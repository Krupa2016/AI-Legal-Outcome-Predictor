import pdfplumber
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

folder_path = "judgments/"
texts = []

print("📄 Extracting text from PDFs...")

for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        with pdfplumber.open(os.path.join(folder_path, file)) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() or ""
            if full_text.strip():
                texts.append(full_text)

print(f"✅ Extracted {len(texts)} documents")

# Convert to vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)

# Save
pickle.dump(vectorizer, open("models/sc_vectorizer.pkl", "wb"))
pickle.dump(X, open("models/sc_vectors.pkl", "wb"))
pickle.dump(texts, open("models/sc_texts.pkl", "wb"))

print("✅ SC dataset processed!")