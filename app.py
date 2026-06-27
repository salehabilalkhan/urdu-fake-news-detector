import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Urdu Fake News Detector", page_icon="📰")

st.title("Urdu Fake News Detector")
st.write("Urdu text likho aur check karo.")

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="facebook/roberta-hate-speech-dynabench-r4-target"
    )

classifier = load_model()

user_input = st.text_area("News Yahan Likhein:", height=150)

if st.button("Check Karo"):
    if user_input.strip() == "":
        st.warning("Kuch likho pehle!")
    else:
        with st.spinner("Check ho raha hai..."):
            result = classifier(user_input)[0]
            score = round(result["score"] * 100, 2)
        st.info("Result: " + result["label"] + " - " + str(score) + "%")