
import os
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate 
from langchain.vectorstores import FAISS 
from langchain.chains import RetrievalQA
import shutil



def load_and_process_legal_documents(pdf_folder_path='doc'):
 """Load and process legal documents, chunk them into smaller segments."""
 documents = []
 for file in os.listdir(pdf_folder_path):
 if file.endswith('.docx'):
 pdf_path = os.path.join(pdf_folder_path, file)
 loader = Docx2txtLoader(pdf_path) 
 documents.extend(loader.load())
 
 
 text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
 chunks = text_splitter.split_documents(documents)
 return chunks



def save_to_faiss_vector_store(document_chunks, db_path="faiss_db"):
 """Save the document chunks into FAISS vector store."""
 if os.path.exists(db_path):
 shutil.rmtree(db_path)
 
 embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
 faiss_vector_store = FAISS.from_documents(document_chunks, embeddings)
 faiss_vector_store.save_local(db_path)
 
 return faiss_vector_store


def load_faiss_vector_store(db_directory_path="faiss_db"):
 """Load the FAISS vector store and return the retriever."""
 embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
 vector_store = FAISS.load_local(db_directory_path, embeddings, allow_dangerous_deserialization=True)
 retriever = vector_store.as_retriever() 
 print(f"Loaded FAISS vector store with {vector_store.index.ntotal} documents.")
 
 return retriever


def create_retrieval_chain_with_llm(retriever):
 """Create a retrieval chain using Ollama LLM."""
 system_prompt = (
 "You are support Assisteance"
 "Use the given context to answer the question. "
 "If you don't know the answer, say you don't know. "
 "Use three sentence maximum and keep the answer concise. "
 "Context: {context}"
 )
 
 prompt = ChatPromptTemplate.from_messages([
 ("system", system_prompt),
 ("human", "{input}"),
 ])
 
 llm = OllamaLLM(model="llama3") 
 question_answer_chain = RetrievalQA.from_chain_type(
 llm, chain_type="stuff", retriever=retriever, return_source_documents=True
 )
 
 return question_answer_chain


def answer_query(chain, query):
 """Retrieve relevant information and generate a legal answer."""
 response = chain.invoke({"query": query})
 answer = response.get("result", "Sorry, I couldn't process that.")
 
 return answer

def terminal_chatbot():
 """Start an interactive terminal-based session for answering legal queries."""
 print("Welcome to the Legal Chatbot! How can I assist you today?")
 
 legal_chunks = load_and_process_legal_documents('doc') 
 faiss_vector_store = save_to_faiss_vector_store(legal_chunks)
 retriever = faiss_vector_store.as_retriever()
 chain = create_retrieval_chain_with_llm(retriever)
 conversation_history = ""

 
 while True:
 user_query = input("\nYou: ")

 if user_query.lower() in ["exit", "quit", "bye"]: 
 print("Goodbye! Stay informed about your legal rights.")
 break
 
 conversation_history += f"You: {user_query}\n"
 answer = answer_query(chain, user_query)
 conversation_history += f"Bot: {answer}\n"
 
 print("Bot: " + answer)



if __name__ == "__main__":
 terminal_chatbot()