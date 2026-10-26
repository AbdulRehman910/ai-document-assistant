# AI Document Assistant Using RAG

An AI-powered document question-answering application that allows users to upload PDF files and ask questions about their content using Retrieval-Augmented Generation (RAG).

## Project Overview

The AI Document Assistant extracts text from uploaded PDF documents, divides the text into smaller chunks, generates embeddings, and retrieves the most relevant passage for a user's question. Groq's language model then generates an answer based on the retrieved context.

The application provides a simple interactive interface built with Streamlit.

## Features

- Upload PDF documents through a web interface.
- Extract and split PDF text into chunks.
- Generate text embeddings using Sentence Transformers.
- Perform semantic search to retrieve the most relevant passage.
- Generate contextual answers using the Groq API.
- Display relevant source passages and PDF page numbers.
- Cache document processing results for repeated use.
- Ask multiple questions in one session.
- Clear chat history or reset the current document.

## Technologies Used

- Python
- Streamlit
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- `all-MiniLM-L6-v2` embedding model
- NumPy
- PyMuPDF
- Groq API
- Python-dotenv

## Project Architecture

1. **PDF Upload:** The user uploads a PDF document.
2. **Text Extraction:** PyMuPDF extracts text from the document.
3. **Text Chunking:** The extracted text is divided into smaller sections.
4. **Embedding Generation:** Sentence Transformers converts each chunk into a numerical vector.
5. **Semantic Retrieval:** The question is embedded and compared with document embeddings.
6. **Answer Generation:** The most relevant passage is sent to the Groq language model.
7. **Response Display:** The application displays the generated answer and source passage.

## Project Structure

```text
AiDocumentAssistant/
├── app.py
├── rag.py
├── build_index.py
├── text_splitter.py
├── search.py
├── test_groq.py
├── documents/
│   └── Machine_Learning_Guide.pdf
├── data/
│   └── document_index.pkl
├── .env
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AiDocumentAssistant
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install streamlit groq python-dotenv pymupdf numpy sentence-transformers
```

### 4. Configure your Groq API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own API key. Never publish your actual API key.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in your terminal, usually `http://localhost:8501`.

## Future Improvements

- Support additional document formats.
- Improve retrieval accuracy with configurable chunk sizes and relevance thresholds.
- Add automated tests and error handling.
- Improve document indexing and persistent caching.
- Deploy the application for public access.

## Learning Outcomes

This project demonstrates practical experience with Python, embeddings, semantic search, LLM integration, prompt design, and building a user-facing AI application using RAG.

## License

Choose and add an appropriate open-source license before distributing the project.
