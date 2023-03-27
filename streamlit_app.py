import streamlit as st
from transformers import pipeline
import openai
from PIL import Image
openai.organization = st.secrets["openai_organization_key"]
openai.api_key = st.secrets["openai_key"]
#openai.Model.list()
pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")

st.title("Hot Dog? Or Not?")
st.write(openai.Model.list())
file_name = st.file_uploader("Upload a hot dog candidate image")

if file_name is not None:
    col1, col2 = st.columns(2)

    image = Image.open(file_name)
    col1.image(image, use_column_width=True)
    predictions = pipeline(image)

    col2.header("Probabilities")
    for p in predictions:
        col2.subheader(f"{ p['label'] }: { round(p['score'] * 100, 1)}%")
