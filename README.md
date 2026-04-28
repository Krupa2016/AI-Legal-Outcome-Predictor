# ⚖️ Legal AI Dashboard  
### AI-Based Legal Case Outcome Prediction & Similar Case Retrieval System

---

## 🚀 Overview

Legal AI Dashboard is an intelligent decision-support system that analyzes legal case descriptions and predicts the likely outcome (**Win/Lose**) using Machine Learning.

The system also retrieves **semantically similar past cases**, highlights **key influencing factors**, and generates **case insights** to assist users in understanding legal scenarios.

It combines **Natural Language Processing (NLP)**, **Machine Learning (ML)**, and **Semantic Search** with a modern dashboard interface.

---

## 🧠 Key Features

### 🔮 Outcome Prediction
- Predicts case result: **Win / Lose**
- Provides confidence score
- Built using Logistic Regression + TF-IDF

---

### 🔍 Explainable AI (XAI)
- Displays **top influencing keywords**
- Helps users understand *why* a prediction was made

---

### 📚 Semantic Similar Case Retrieval
- Uses **Sentence-BERT embeddings**
- Finds contextually similar legal cases (not just keyword match)

---

### 🧾 Case Summarization
- Extracts meaningful insights from long legal texts
- Removes noisy data (URLs, headers)
- Displays clean summaries

---

### 📊 Interactive Dashboard
- Navy blue legal-themed UI
- Confidence visualization using Chart.js
- Card-based layout
- Responsive design

---

### 💡 AI Suggestions
- Provides basic legal strategy hints:
  - Strong case → proceed  
  - Weak case → consider settlement  

---

## 🛠️ Tech Stack

| Layer | Technology |
|------|-----------|
| Backend | Flask (Python) |
| ML/NLP | Scikit-learn, Sentence-Transformers |
| Data Processing | Pandas, NumPy, PDFPlumber |
| Frontend | HTML, CSS, JavaScript |
| Visualization | Chart.js |

---
