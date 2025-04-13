# legal_chatbot.py

import os
import shutil
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


def load_and_process_legal_documents(folder_path='doc'):
    """Load .docx legal documents and chunk them."""
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith('.docx'):
            doc_path = os.path.join(folder_path, file)
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def save_to_faiss_vector_store(document_chunks, db_path="faiss_db"):
    """Save document chunks to FAISS vector store."""
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': "cpu"}
    )
    faiss_store = FAISS.from_documents(document_chunks, embeddings)
    faiss_store.save_local(db_path)
    return faiss_store


def load_faiss_vector_store(db_path="faiss_db"):
    """Load FAISS vector store and return retriever."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': "cpu"}
    )
    vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
    print(f"[INFO] Loaded FAISS store with {vector_store.index.ntotal} documents.")
    return retriever


def create_retrieval_chain_with_llm(retriever):
    """Create RetrievalQA chain with Ollama LLM."""
    system_prompt = (
        "You are a legal assistant. "
        "Use the context to answer legal questions concisely in three sentences or less. "
        "If unsure, say you don't know. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    llm = OllamaLLM(model="llama3")
    return RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )


def answer_query(chain, query):
    """Answer a user query using the LLM chain."""
    response = chain.invoke({"query": query})
    return response.get("result", "Sorry, I couldn't process that.")


def terminal_chatbot():
    """Run terminal-based chatbot session."""
    print("🤖 Welcome to the Legal Chatbot! Type your question or 'exit' to quit.")

    chunks = load_and_process_legal_documents('doc')
    faiss_store = save_to_faiss_vector_store(chunks)
    retriever = faiss_store.as_retriever()
    chain = create_retrieval_chain_with_llm(retriever)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye! Stay informed about your legal rights.")
            break

        answer = answer_query(chain, user_input)
        print("Agent:", answer)


if __name__ == "__main__":
    terminal_chatbot()
