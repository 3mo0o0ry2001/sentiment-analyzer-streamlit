import streamlit as st
import joblib
import numpy as np

# 🧠 تحميل الموديل والـvectorizer
@st.cache_resource
def load_model():
    model = joblib.load("data/sentiment_model_imdb.pkl")
    vectorizer = joblib.load("data/tfidf_vectorizer_imdb.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# 🎨 إعداد الصفحة
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="centered")

st.title("🎬 IMDB Sentiment Analyzer")
st.markdown(
    "<p style='font-size:18px;'>اكتب أي جملة بالإنجليزية، والموديل هيحدد المشاعر مع نسبة الثقة.</p>",
    unsafe_allow_html=True
)
st.write("---")

# ✍️ إدخال المستخدم
user_input = st.text_area("🗣️ Write your sentence here:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        # تحويل النص إلى تمثيل رقمي
        X_input = vectorizer.transform([user_input])

        # التوقع ونسبة الثقة
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        confidence = np.max(probabilities) * 100

        # 🧩 لون النتيجة بناءً على الثقة
        if confidence >= 80:
            color = "green"
        elif confidence >= 60:
            color = "orange"
        else:
            color = "red"

        # 🧠 عرض النتيجة
        st.markdown(
            f"<h3>🧠 Predicted Sentiment: "
            f"<span style='color:{'limegreen' if prediction=='positive' else 'red'};'>"
            f"{prediction.upper()}</span></h3>",
            unsafe_allow_html=True
        )
        st.markdown(f"<h4>📊 Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)

        # 🎚️ شريط الثقة
        st.progress(int(confidence))

        # 💬 تعليق بسيط حسب النتيجة
        if prediction == "positive":
            st.success("😄 This seems to be a **positive** review!")
        else:
            st.error("😡 This seems to be a **negative** review!")

        # 🔎 تعليق إضافي حسب الثقة
        if confidence < 60:
            st.info("⚠️ Low confidence — model is uncertain about this prediction.")
