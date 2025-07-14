# LangChain RAG Experimental Project

This project demonstrates a simple Retrieval-Augmented Generation (RAG) workflow using Google Gemini embeddings and Chroma vector store, powered by LangChain. It shows how to embed documents, store them in a vector database, and perform similarity-based retrieval for question answering.

## Features

- Uses Google Generative AI (Gemini) for text embeddings
- Stores and retrieves documents using Chroma vector store
- Demonstrates similarity search for RAG
- Includes clear print statements for each step

## Setup

### 1. Clone the Repository

```
git clone <repo-url>
cd gemini-langchain-rag
```

### 2. Install Dependencies

Make sure you have Python 3.8+ and `pip` installed. Then run:

```
pip install -r requirements.txt
```

### 3. Store Your Google API Key in a `.env` File

Create a file named `.env` in the `gemini-langchain-rag` directory with the following content:

```
GOOGLE_API_KEY=your-api-key-here
```

Replace `your-api-key-here` with your actual API key.

**Do NOT hardcode your API key in the script for security reasons.**

## Usage

Run the script:

```
python main.py
```
