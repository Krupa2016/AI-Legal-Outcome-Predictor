from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# -------------------------
# Load prediction model
# -------------------------
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# -------------------------
# Load similarity data
# -------------------------
sc_texts = pickle.load(open("models/sc_texts.pkl", "rb"))

# 🔥 Semantic model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings
sc_embeddings = semantic_model.encode(sc_texts, show_progress_bar=True)

# -------------------------
# Explainability (keywords)
# -------------------------
def get_top_keywords(input_text):
    vec = vectorizer.transform([input_text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    scores = vec.toarray()[0] * coefs
    top_indices = np.argsort(scores)[-5:][::-1]

    return [feature_names[i] for i in top_indices if scores[i] != 0]

# -------------------------
# Summarization
# -------------------------
def summarize(text):
    sentences = text.split(".")

    clean_sentences = []

    for s in sentences:
        s = s.strip()

        # remove junk lines
        if len(s) < 40:
            continue
        if "http" in s.lower():
            continue
        if "judis" in s.lower():
            continue

        clean_sentences.append(s)

    if len(clean_sentences) == 0:
        return "No meaningful summary available."

    return ". ".join(clean_sentences[:2]) + "."

# -------------------------
# Semantic Similarity
# -------------------------
def get_similar_cases(input_text):
    input_embedding = semantic_model.encode([input_text])
    sims = cosine_similarity(input_embedding, sc_embeddings)

    top_indices = np.argsort(sims[0])[-3:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "summary": summarize(sc_texts[i])
        })

    return results

# -------------------------
# API
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    input_text = request.json["text"]

    # Prediction
    vec = vectorizer.transform([input_text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]

    # Explainability
    keywords = get_top_keywords(input_text)

    # Similar cases
    similar_cases = get_similar_cases(input_text)

    return jsonify({
        "prediction": "Win" if pred == 1 else "Lose",
        "confidence": float(prob),
        "keywords": keywords,
        "similar_cases": similar_cases
    })


@app.route("/")
def home():
    return "🚀 Legal AI API running with XAI + Semantic Search"


if __name__ == "__main__":
    app.run(debug=True)