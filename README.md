# sampleset

# Financial QA Chatbot
This repository contains a Retrieval-Augmented Generation (RAG)-based chatbot built using Streamlit. The chatbot processes financial PDFs (such as Profit & Loss statements) and answers user queries based on the content of the uploaded document. It leverages PyMuPDF for text extraction, rake-nltk for keyword extraction, sentence-transformers for embeddings, and transformers for generating responses.

# Features
Upload Financial PDFs: Users can upload financial documents (e.g., balance sheets or P&L statements).

Query-Based Retrieval: The chatbot identifies relevant sections of the document based on user queries.

Contextual Answers: It generates answers using GPT-based models like EleutherAI/gpt-neo-1.3B.

Interactive UI: A user-friendly interface built with Streamlit for seamless interaction.

Keyword Extraction: Extracts key phrases using rake-nltk to find relevant content.

Semantic Search: Uses embeddings from sentence-transformers to find the most relevant chunks.

# Tech Stack
## Libraries
Streamlit: For creating the web app interface.

PyMuPDF (pymupdf): For extracting text from PDFs.

RAKE (rake-nltk): For keyword extraction.

Sentence-Transformers: For generating embeddings of text chunks.

Transformers (Hugging Face): For GPT-based text generation.

NumPy & SciPy: For cosine similarity computations.

# Usage
After installing the dependencies in your environment, run the part2.py file. It will open the Streamlit app in your browser at http://localhost:8501 by default.
Upload a financial PDF document (e.g., a P&L statement or balance sheet).
Enter a financial query in the text input box (e.g., "What is the total revenue for Q3 2024?").
View the chatbot's answer along with the relevant context extracted from the PDF.

# Dependencies
The following libraries are used in this project:
streamlit==1.41.1

pymupdf==1.25.2

sentence-transformers==3.3.1

transformers==4.48.1

rake-nltk==1.0.6

nltk

numpy

scipy

pandas

torch
