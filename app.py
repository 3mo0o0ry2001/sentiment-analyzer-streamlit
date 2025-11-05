import streamlit as st
import joblib
import numpy as np

# 🧠 تحميل الموديل
@st.cache_resource
def load_model():
    model = joblib.load("data/sentiment_model_v2.pkl")
    vectorizer = joblib.load("data/tfidf_vectorizer_v2.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# 🎨 إعداد الصفحة
st.set_page_config(page_title="Sentiment Analyzer v2", page_icon="🧠", layout="centered")

# 👋 الهيدر
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🧠 Sentiment Analyzer v2</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:17px;'>Analyze English sentences and discover their sentiment instantly.</p>", unsafe_allow_html=True)
st.write("---")

# ✍️ إدخال المستخدم
user_input = st.text_area("🗣️ Write your sentence here:", placeholder="Type your review or opinion...")

if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a sentence.")
    else:
        X_input = vectorizer.transform([user_input.lower()])
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        confidence = np.max(probabilities) * 100

        sentiment_label = "Positive 😄" if prediction == 1 else "Negative 😡"
        color = "#4CAF50" if prediction == 1 else "#FF4B4B"

        # 🎯 النتيجة
        st.markdown(f"<h3 style='color:{color}; text-align:center;'>🧠 {sentiment_label}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align:center;'>📊 Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)
        st.progress(int(confidence))

        # 💬 ملاحظات
        if confidence < 60:
            st.info("⚠️ Low confidence — model is uncertain about this prediction.")

st.write("---")
st.caption("Developed by Omar Ayoub | Version 2.0 | Powered by Logistic Regression + TF-IDF")
