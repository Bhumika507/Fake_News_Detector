import streamlit as st
import joblib
import re

# 1. LOAD SAVED ASSETS (Must match filenames in your training code)
model = joblib.load('model.joblib')
vectorize = joblib.load('vectorize.joblib')

# 2. PREPROCESSING FUNCTION (Must match your training logic)
def clear_title(title):
    title = title.lower()
    title = re.sub("\n", "", title)
    return title

# 3. UI DESIGN (Streamlit)
st.set_page_config(page_title="Fake News Detector")
st.title("🔍 Fake News Detector")
st.markdown("---")
st.write("Enter a news article title below to check whether it is **Fake** or **Real**.")

user_input = st.text_area("News Title to Analyze:", placeholder="Paste title here...", height=150)

if st.button("Check"):
    if user_input.strip() != "":
        cleaned_text = clear_title(user_input)
        vectorized_text = vectorize.transform([cleaned_text])
        prediction = model.predict(vectorized_text)

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success("✅ This news is **REAL**")
        else:
            st.error("🚨 This news is **FAKE**")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
