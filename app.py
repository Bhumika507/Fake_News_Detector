import streamlit as st
import joblib
import re

st.set_page_config(page_title="Fake News Detector")

# 1. LOAD SAVED ASSETS
try:
    model = joblib.load('model.joblib')
    vectorize = joblib.load('vectorize.joblib')
except FileNotFoundError:
    st.title("Fake News Detector")
    st.error("Required model files not found: model.joblib and vectorize.joblib.")
    st.info("Run the training script first to generate these files.")
    st.stop()
except Exception as exc:
    st.title("Fake News Detector")
    st.error(f"Unable to load saved model files: {exc}")
    st.stop()

# 2. PREPROCESSING FUNCTION

def clear_title(title):
    """Clean the input title before prediction."""
    title = title.lower()
    title = re.sub(r"\n", "", title)
    return title

# 3. UI DESIGN
st.set_page_config(page_title="Fake News Detector")

st.title("Fake News Detector")
st.markdown("---")
st.write("Enter a news article title below to check whether it is **Fake** or **Real**.")

user_input = st.text_area(
    "News Title to Analyze:",
    placeholder="Paste title here...",
    height=150,
)

if st.button("Check"):
    if user_input.strip() != "":
        cleaned_text = clear_title(user_input)
        vectorized_text = vectorize.transform([cleaned_text])
        prediction = model.predict(vectorized_text)

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success("This news is **REAL**")
        else:
            st.error("This news is **FAKE**")
    else:
        st.warning("Please enter some text to analyze.")
