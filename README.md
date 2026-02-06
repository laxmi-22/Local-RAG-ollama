# 📄 Local RAG using Ollama

(Part of My RAG Learning Journey – Post 3)

## 📌 Context

This repository is part of my RAG learning journey, where I’m exploring different ways to design and implement Retrieval-Augmented Generation (RAG) systems.

In this phase, I focused on answering a practical question:

Can we build a complete RAG pipeline locally — without cloud APIs — while still maintaining retrieval quality and grounding?

This project demonstrates a Local RAG system using Ollama, designed to understand privacy, cost, and control trade-offs in real-world scenarios.

## 🧠 What This Project Demonstrates

- End-to-end local RAG pipeline
- PDF-based question answering
- Retrieval grounded strictly in document context
- Focus on preprocessing and retrieval quality, not UI

## 🏗️ High-Level Flow

PDF Document
   ↓
Preprocessing (remove headers, footers, emojis)
   ↓
Text Chunking
   ↓
Embeddings (local – Ollama)
   ↓
Vector Store (FAISS)
   ↓
Retriever (MMR)
   ↓
LLM (Ollama)
   ↓
Context-grounded Answer

## 🔍 Key Learning (Core Insight)

Initially, I directly moved from PDF loading to chunking.
Through iteration, I realized that preprocessing PDFs before chunking — especially removing repetitive headers, footers, and emojis — significantly improves:

- Embedding quality
- Retrieval relevance
- Final answer accuracy

## ➡️ Cleaner input → better retrieval → more reliable RAG output

This learning influenced the final design of this pipeline.

## 🔧 Design Choices & Reasoning
## ✔ PDF Preprocessing

- Removes repetitive headers & footers
- Removes emojis and noise
- Reduces embedding pollution
- Improves semantic similarity during retrieval

## ✔ Chunking Strategy

- Recursive character-based splitting
- Chunk overlap to preserve context
- Page metadata retained

## ✔ Vector Store

FAISS (local, lightweight, fast)

## ✔ Retriever

## - MMR (Max Marginal Relevance) used to:

- Reduce redundant chunks
- Improve diversity in retrieved context
- Balance relevance vs overlap

## ✔ Strict Grounding Prompt

The LLM is instructed to:

- Use only retrieved context
- Avoid hallucinations
- Respond with “I don’t know” when information is missing

## 🛠️ Tech Stack

- Python
- LangChain
- Ollama
- FAISS
- PyMuPDF
- Local LLMs (e.g., Llama3 / Mistral)

## ⚙️ Setup Instructions
## 1️⃣ Install Ollama

Download from:
https://ollama.com

Verify:
ollama --version

## 2️⃣ Pull a Local Model
ollama pull llama3

(You can replace with mistral or other supported models.)

## 3️⃣ Create Virtual Environment
python -m venv venv

source venv/bin/activate     # Windows: venv\Scripts\activate

## 4️⃣ Install Dependencies
pip install -r requirements.txt

(Streamlit is included for future UI extension but not used in this version.)

## 5️⃣ Configure PDF Path
place PDF file in data folder 

Update in main():
pdf_path = "data/filename.pdf"

## 6️⃣ Run the Application
python app.py

Ask questions in the terminal.
Type exit to quit.

## ⚠️ Limitations

- Performance depends on local hardware
- Local models may have weaker reasoning than cloud LLMs
- Context window limitations
- CLI-based interaction (no UI)

## 🚀 Future Enhancements

- Streamlit UI for interactive chat
- Support for multiple PDFs
- Persistent vector store
- Hybrid retrieval (BM25 + embeddings)
- Embedding caching
- Agentic RAG extensions

## 🎯 Why No UI?

The UI layer is intentionally skipped to keep focus on:

- RAG architecture
- Retrieval quality
- Preprocessing impact
- Grounding strategies
UI can be added easily once the pipeline is solid.

## 🤝 Closing Note

This project reflects hands-on learning and iteration, not a demo-first approach.
It is part of my broader effort to deeply understand how RAG systems behave in real conditions, especially when built locally.

Feedback and discussions are welcome.


