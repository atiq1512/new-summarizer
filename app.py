import streamlit as st
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

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
    Paste a news article below to get both **extractive** and **abstractive** summaries.
    - **Extractive Summary**: Highlights key sentences from the original text.
    - **Abstractive Summary**: Generates a concise paragraph in new words.
    """
)

# User input
article_text = st.text_area("Enter Article Here", height=300)

if st.button("Generate Summary"):
    if article_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        # -----------------------------
        # Extractive Summary
        # -----------------------------
        parser = PlaintextParser.from_string(article_text, Tokenizer("english"))
        extractive_summary = " ".join([str(s) for s in extractive_summarizer(parser.document, 3)])

        # -----------------------------
        # Abstractive Summary
        # -----------------------------
        abstractive_summary = abstractive_summarizer(
            article_text, max_length=150, min_length=50, do_sample=False
        )[0]['summary_text']

        # -----------------------------
        # Display Results
        # -----------------------------
        st.subheader("📝 Extractive Summary (Key Sentences)")
        st.write(extractive_summary)

        st.subheader("✍️ Abstractive Summary (Paragraph in New Words)")
        st.write(abstractive_summary)

        # Optional: Save summaries as downloadable files
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
