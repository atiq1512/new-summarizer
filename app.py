import streamlit as st
import nltk
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# -----------------------------
# Download NLTK data
# -----------------------------
nltk.download('punkt')

# -----------------------------
# Load Models
# -----------------------------
@st.cache(allow_output_mutation=True)
def load_models():
    abstractive_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    extractive_model = LexRankSummarizer()
    return abstractive_model, extractive_model

abstractive_summarizer, extractive_summarizer = load_models()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📰 News Summarizer System")
st.write(
    """
    Paste a news article or upload a `.txt` file to get both **extractive** and **abstractive** summaries.
    - **Extractive Summary**: Highlights key sentences from the original text.
    - **Abstractive Summary**: Generates a concise paragraph in new words.
    """
)

# -----------------------------
# User input: Paste or Upload
# -----------------------------
article_text = ""

# File uploader
uploaded_file = st.file_uploader("Upload a news article (.txt)", type="txt")
if uploaded_file is not None:
    article_text = uploaded_file.read().decode("utf-8")

# Text area for manual paste
manual_text = st.text_area("Or paste your article here:", height=300)
if manual_text.strip() != "":
    article_text = manual_text

# -----------------------------
# Generate Summary
# -----------------------------
if st.button("Generate Summary"):
    if article_text.strip() == "":
        st.warning("Please provide an article by pasting text or uploading a file!")
    else:
        # Extractive summary
        parser = PlaintextParser.from_string(article_text, Tokenizer("english"))
        extractive_summary = " ".join([str(s) for s in extractive_summarizer(parser.document, 3)])

        # Abstractive summary
        abstractive_summary = abstractive_summarizer(
            article_text, max_length=150, min_length=50, do_sample=False
        )[0]['summary_text']

        # Display results
        st.subheader("📝 Extractive Summary (Key Sentences)")
        st.write(extractive_summary)

        st.subheader("✍️ Abstractive Summary (Paragraph in New Words)")
        st.write(abstractive_summary)

        # Download buttons
        st.download_button(
            label="Download Extractive Summary",
            data=extractive_summary,
            file_name="extractive_summary.txt",
            mime="text/plain"
        )
        st.download_button(
            label="Download Abstractive Summary",
            data=abstractive_summary,
            file_name="abstractive_summary.txt",
            mime="text/plain"
        )
