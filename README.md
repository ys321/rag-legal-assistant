# RAG Legal Assistant

A Retrieval-Augmented Generation (RAG) based chatbot that helps answer legal queries by extracting relevant information from `.docx` legal documents. Built using **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Ollama LLM**.

![Github](https://github.com/user-attachments/assets/bce6d87e-ce5b-4f8d-b06e-d8ee261e5b12)

---

## 🧠 Features

- Upload and process `.docx` legal documents
- Automatically chunk and embed documents using Sentence Transformers
- Store embeddings in a FAISS vector store
- Use LLM (e.g., LLaMA 3 via Ollama) to answer questions
- Terminal-based interactive Q&A chatbot

---

## Requirements

### Hardware Requirements:
- **GPU**: A GPU with at least **4 GB of memory** is required for optimal performance. The model may run into memory issues if the available GPU memory is less than this.
---

## 🛠 Tech Stack

- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Transformers](https://huggingface.co/sentence-transformers)
- [Ollama](https://ollama.com/) (e.g., LLaMA 3)
- Python 3.10+

---

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ys321/rag-legal-assistant.git
cd rag-legal-assistant
```

### 2. Create a Virtual Environment

#### On **Linux/macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On **Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and Run Ollama
Make sure you have [Ollama](https://ollama.com/) installed and running.

Then pull a model (e.g., LLaMA 3):
```bash
ollama pull llama3
```

### 5. Add Legal `.docx` Files
Put your `.docx` files inside the `doc/` directory.

---

## 🚀 Run the Legal Chatbot
```bash
python app.py
```

Then simply start chatting with your legal assistant:
```
Welcome to the Legal Chatbot! How can I assist you today?
You: What are the terms for payment in the agreement?
Agent: The agreement states that INR 1,50,000 is paid upon signing and INR 3,50,000 on delivery.
```

---

## 📁 Project Structure
```
rag-legal-assistant/
├── app.py                      # Main entry point
├── doc/                        # Directory for `.docx` files
├── faiss_db/                   # Auto-generated FAISS DB (after run)
└── requirements.txt            # Python dependencies
```

---

## ✅ To-Do / Enhancements
- Web interface (Streamlit / Gradio)
- PDF loader support
- LLM selection menu
- History tracking

---
