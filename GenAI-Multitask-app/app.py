import streamlit as st
import os
from huggingface_hub import InferenceClient

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="GenAI MultiTask App", page_icon="🤖")

st.title("🤖 GenAI Multi-Task Application")

# -------------------------------
# Load Hugging Face Client
# -------------------------------
@st.cache_resource
def load_client():
    return InferenceClient(
        provider="hf-inference",
        api_key=st.secrets["HF_TOKEN"]   # ✅ secure (NOT os.environ)
    )

client = load_client()

# -------------------------------
# Task Selection
# -------------------------------
task = st.sidebar.selectbox(
    "Choose a Task",
    (
        "Text Summarization",
        "Text Translation",
        "Text Writing",
        "NER (Entity Recognition)",
        "Sentiment Analysis"
    )
)

# -------------------------------
# Input
# -------------------------------
input_text = st.text_area("Enter your text here:", height=250)

target_lang = ""
if task == "Text Translation":
    target_lang = st.text_input("Enter target language (e.g., French, Spanish):")

# -------------------------------
# Run Button
# -------------------------------
if st.button("🚀 Run Task"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text")

    elif task == "Text Translation" and not target_lang.strip():
        st.warning("⚠️ Please enter target language")

    else:
        try:
            st.info("Processing... ⏳")

            # -------------------------------
            # TASK HANDLING
            # -------------------------------
            if task == "Text Summarization":
                result = client.summarization(
                    input_text,
                    model="facebook/bart-large-cnn"
                )
                output = result

            elif task == "Text Translation":
                result = client.text_generation(
                    f"Translate to {target_lang}: {input_text}",
                    model="tencent/HY-MT1.5-1.8B",
                    max_new_tokens=200
                )
                output = result

            elif task == "Text Writing":
                result = client.text_generation(
                    f"Write a detailed article about: {input_text}",
                    model="google/flan-t5-base",
                    max_new_tokens=300
                )
                output = result

            elif task == "NER (Entity Recognition)":
                result = client.token_classification(
                    input_text,
                    model="dslim/bert-base-NER"
                )
                output = "\n".join([f"{ent['word']} → {ent['entity_group']}" for ent in result])

            elif task == "Sentiment Analysis":
                result = client.text_classification(
                    input_text,
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                output = result[0]["label"]

            # -------------------------------
            # Display Output
            # -------------------------------
            st.success("✅ Result")
            st.write(output)

        except Exception as e:
            st.error(f"❌ Error: {e}")
