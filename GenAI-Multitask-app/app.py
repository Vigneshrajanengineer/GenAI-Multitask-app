import streamlit as st
import requests

st.set_page_config(page_title="GenAI MultiTask App", page_icon=":robot_face:")

st.title("GenAI Multi-Task Application")

task = st.sidebar.selectbox(
    "Choose a Task",
    ("Text Summarization", "Text Translation", "Text Writing", "NER (Entity Recognition)", "Sentiment Analysis")
)

input_text = st.text_area("Enter your text here:", height=300)

additional_input = None
if task == "Text Translation":
    additional_input = st.text_input("Enter target language (e.g., French, Spanish):")

if st.button("Run Task"):
    if input_text:
        if task == "Text Summarization":
            model = "summarizer-llm"
            prompt = f"Summarize the following text:\n\n{input_text}"
        elif task == "Text Translation":
            model = "translator-llm"
            prompt = f"Translate this text into {additional_input}:\n\n{input_text}"
        elif task == "Text Writing":
            model = "writer-llm"
            prompt = f"Write a creative piece based on:\n\n{input_text}"
        elif task == "NER (Entity Recognition)":
            model = "ner-llm"
            prompt = f"Extract and list named entities (person, organization, location, etc.) from this text:\n\n{input_text}"
        elif task == "Sentiment Analysis":
            model = "sentiment-llm"
            prompt = f"Analyze the sentiment of this text. Reply with Positive, Negative, or Neutral:\n\n{input_text}"
        
        try:
            response = requests.post( f"https://api-inference.huggingface.co/models/{model}", headers={"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"},
    json={"inputs": prompt})
            result = response.json()['response']
            st.success(result)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter some text.")
