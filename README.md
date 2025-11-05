# 🧠 Sentiment Analyzer v2

A simple yet powerful **Sentiment Analysis App** built with **Streamlit**, trained on real-world datasets (IMDB, Amazon, Yelp).  
It predicts whether a sentence expresses a positive or negative sentiment — instantly.

🌐 **Live Demo:** [Try it on Streamlit Cloud](https://sentiment-analyzer-app-mg22nggsfrgp4bu7qdf8bz.streamlit.app/)

---

## 🚀 Features
- Classifies English text as **Positive 😄** or **Negative 😡**
- Uses **TF-IDF** + **Logistic Regression**
- Shows confidence percentage with a progress bar
- Deployed online via **Streamlit Cloud**
- Optimized caching for fast inference

---

## 📂 Project Structure
sentiment-analyzer-streamlit/
│
├── app.py # Streamlit UI
├── data/
│ ├── sentiment_model_v2.pkl
│ ├── tfidf_vectorizer_v2.pkl
│
├── requirements.txt # Dependencies
└── README.md # Project documentation

---

## 🧠 Model Details
Trained on combined datasets:
- IMDB movie reviews
- Amazon product feedback
- Yelp restaurant comments

Accuracy achieved: **~79%**

---

## ⚙️ Tech Stack
- Python 3.13
- Streamlit
- Scikit-learn
- Pandas, NumPy, Joblib

---

## 👨‍💻 Developer
**Omar Ayoub**  
AI / NLP Engineer — Passionate about creating practical AI tools.  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/omarayoubai/)
