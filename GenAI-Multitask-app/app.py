import streamlit as st
import requests

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="GenAI MultiTask App", page_icon="🤖")

st.title("🤖 GenAI Multi-Task Application")

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
# Model Mapping (REAL MODELS)
# -------------------------------
MODEL_MAP = {
    "Text Summarization": "facebook/bart-large-cnn",
    "Text Translation": "google/flan-t5-base",
    "Text Writing": "google/flan-t5-base",
    "NER (Entity Recognition)": "dslim/bert-base-NER",
    "Sentiment Analysis": "distilbert-base-uncased-finetuned-sst-2-english"
}

# -------------------------------
# API Call Function
# -------------------------------
def query(model, payload):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {st.secrets['HF_TOKEN']}"
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# -------------------------------
# Run Button
# -------------------------------
if st.button("🚀 Run Task"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text")

    elif task == "Text Translation" and not target_lang.strip():
        st.warning("⚠️ Please enter target language")

    else:
        with st.spinner("Processing... ⏳"):
            try:
                model = MODEL_MAP[task]

                # -------------------------------
                # Payload Creation
                # -------------------------------
                if task == "Text Summarization":
                    payload = {"inputs": input_text}

                elif task == "Text Translation":
                    payload = {"inputs": f"Translate to {target_lang}: {input_text}"}

                elif task == "Text Writing":
                    payload = {"inputs": f"Write a detailed content about: {input_text}"}

                elif task == "NER (Entity Recognition)":
                    payload = {"inputs": input_text}

                elif task == "Sentiment Analysis":
                    payload = {"inputs": input_text}

                # -------------------------------
                # API Call
                # -------------------------------
                output = query(model, payload)

                # -------------------------------
                # Handle Model Loading / Errors
                # -------------------------------
                if isinstance(output, dict) and "error" in output:
                    st.warning("⚠️ Model is loading... Please try again in a few seconds.")
                    st.stop()

                st.success("✅ Result")

                # -------------------------------
                # Output Handling
                # -------------------------------
                if task == "Text Summarization":
                    st.write(output[0]["summary_text"])

                elif task in ["Text Translation", "Text Writing"]:
                    st.write(output[0]["generated_text"])

                elif task == "NER (Entity Recognition)":
                    for ent in output:
                        st.write(f"{ent['word']} → {ent['entity_group']}")

                elif task == "Sentiment Analysis":
                    st.write(f"Sentiment: {output[0]['label']}")

            except Exception as e:
                st.error(f"❌ Error: {e}")
