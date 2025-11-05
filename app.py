import streamlit as st
import joblib
import numpy as np

# 🧠 تحميل الموديل والـvectorizer (الإصدار الجديد)
@st.cache_resource
def load_model():
    model = joblib.load("data/sentiment_model_v2.pkl")
    vectorizer = joblib.load("data/tfidf_vectorizer_v2.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# 🎨 إعداد الصفحة
st.set_page_config(page_title="Sentiment Analyzer v2", page_icon="🧠", layout="centered")

st.title("🧠 Sentiment Analyzer v2")
st.markdown(
    "<p style='font-size:18px;'>اكتب أي جملة بالإنجليزية، والموديل هيحدد إذا كانت إيجابية أو سلبية مع نسبة الثقة.</p>",
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
        X_input = vectorizer.transform([user_input.lower()])

        # التوقع ونسبة الثقة
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        confidence = np.max(probabilities) * 100

        sentiment_label = "Positive" if prediction == 1 else "Negative"
        color = "limegreen" if prediction == 1 else "red"

        # 🧠 عرض النتيجة
        st.markdown(
            f"<h3>🧠 Predicted Sentiment: <span style='color:{color};'>{sentiment_label}</span></h3>",
            unsafe_allow_html=True
        )
        st.markdown(f"<h4>📊 Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)

        # 🎚️ شريط الثقة
        st.progress(int(confidence))

        # 💬 تعليق بسيط حسب النتيجة
        if prediction == 1:
            st.success("😄 This seems to be a **positive** review!")
        else:
            st.error("😡 This seems to be a **negative** review!")

        # 🔎 تعليق إضافي حسب الثقة
        if confidence < 60:
            st.info("⚠️ Low confidence — model is uncertain about this prediction.")
