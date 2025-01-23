import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from rake_nltk import Rake
import numpy as np
import re
from scipy.spatial.distance import cosine
import pandas as pd

import os
os.environ["TORCH_FORCE_LAZY_MODULES"] = "0"

import nltk
nltk.download('stopwords')

# import nltk
nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_model = AutoModelForCausalLM.from_pretrained(model_name)
rake = Rake()

def extract_text_from_pdf(pdf_file):
    pdf_text = {}
    document = fitz.open(pdf_file)  # Open the file using its path
    for page_number in range(document.page_count):
        page = document.load_page(page_number)
        pdf_text[page_number + 1] = page.get_text()
    document.close()
    return pdf_text


def preprocess_pdf_text(pdf_text):
    processed_text = {}
    for page, text in pdf_text.items():
        text = re.sub(r"\n\s*\n", "\n", text.strip())
        relevant_lines = []
        for line in text.split("\n"):
            if re.search(r"\b(balance sheet|revenue|assets|liabilities|profit|income|expense)\b", line, re.IGNORECASE):
                relevant_lines.append(line)
            elif re.search(r"[\d,]+(\.\d+)?", line): 
                relevant_lines.append(line)
        processed_text[page] = "\n".join(relevant_lines)
    return processed_text

def chunk_text(processed_text, chunk_size=1000, chunk_overlap=200):
    chunks = {}
    for page, text in processed_text.items():
        chunk_list = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk_list.append(text[i:i + chunk_size])
        chunks[page] = chunk_list
    return chunks

def get_phrase_embeddings(chunks):
    chunk_phrases = {}
    for page, chunk_list in chunks.items():
        for chunk_number, chunk in enumerate(chunk_list, start=1):
            rake.extract_keywords_from_text(chunk)
            phrases = rake.get_ranked_phrases()
            chunk_phrases[(page, chunk_number)] = [(phrase, embedding_model.encode(phrase)) for phrase in phrases]
    return chunk_phrases

def find_relevant_chunks(query, page_chunks, phrase_embeddings):
    rake.extract_keywords_from_text(query)
    query_phrases = rake.get_ranked_phrases()
    query_embeddings = [embedding_model.encode(phrase) for phrase in query_phrases]

    chunk_similarities = {}
    for (page, chunk_number), phrases in phrase_embeddings.items():
        similarities = []
        for phrase, embedding in phrases:
            phrase_similarities = [1 - cosine(embedding, query_embedding) for query_embedding in query_embeddings]
            similarities.append(max(phrase_similarities))
        chunk_similarities[(page, chunk_number)] = np.mean(similarities)

    # Sort chunks by similarity and take the top 5
    top_chunks = sorted(chunk_similarities.items(), key=lambda x: x[1], reverse=True)[:5]

    # Correctly unpack the tuple
    selected_chunks = [
        page_chunks[int(page)][int(chunk_number) - 1] for ((page, chunk_number), _) in top_chunks
    ]
    return selected_chunks



def generate_response(context, query):
    prompt = f"Answer the following query based on the provided text:\n\n{context}\n\nQuery: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = generation_model.generate(inputs["input_ids"], max_new_tokens=300, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

# Streamlit Interface
st.title("Financial QA Chatbot")
st.write("Upload a financial PDF document and ask questions.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
query = st.text_input("Enter your query:")

if uploaded_file and query:
    st.write("Processing your document...")
    pdf_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = preprocess_pdf_text(pdf_text)
    chunks = chunk_text(cleaned_text)
    chunk_phrases = get_phrase_embeddings(chunks)
    relevant_chunks = find_relevant_chunks(query, chunk_phrases)
    context = "\n\n".join(relevant_chunks)
    answer = generate_response(context, query)

    st.write(f"**Answer:** {answer}")
    st.write("**Relevant Context:**")
    st.write(context)

# import gradio as gr

# def chatbot_interface(pdf_file, query):
#     pdf_text = extract_text_from_pdf(pdf_file)
#     cleaned_text = preprocess_pdf_text(pdf_text)
#     chunks = chunk_text(cleaned_text)
#     chunk_phrases = get_phrase_embeddings(chunks)
#     relevant_chunks = find_relevant_chunks(query, chunks, chunk_phrases)
#     relevant_chunks = find_relevant_chunks(query, chunks, chunk_phrases)
#     context = "\n\n".join(relevant_chunks)
#     answer = generate_response(context, query)
#     return answer, context

# iface = gr.Interface(
#     fn=chatbot_interface,
#     inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Enter Query")],
#     outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Relevant Context")],
#     title="Financial QA Chatbot",
#     description="Upload a financial PDF and ask questions about it."
# )

# iface.launch()

