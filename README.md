# Chatbot with FAISS and LLM  

## 📌 Project Overview  
This project implements a **chatbot** that uses a **vector database (FAISS)** and a **Large Language Model (LLM)** to answer user queries based on uploaded PDF documents.  
The chatbot follows a **RAG (Retrieval-Augmented Generation) approach**, where it retrieves relevant documents from FAISS before generating responses.  

## 🚀 Features  
✅ **Retrieval-based Q&A** – Uses FAISS to find the most relevant documents.  
✅ **LLM-powered responses** – Generates answers using a Large Language Model.  
✅ **Session state management** – Stores chat history in Streamlit.  
✅ **Custom prompts** – Uses a predefined template for more accurate responses.  

## 🛠️ Installation  

### 1️⃣ Clone the repository  
```bash 
git clone https://github.com/albamvv/chatbot.git
```
```bash 
cd chatbot
```

### 2️⃣ Create and activate a virtual environment
```bash  
python -m venv env 
```
```bash
python3 -m venv env
```

```bash 
source env/bin/activate  # En Linux/macOS
```

```bash
env\Scripts\activate  # En Windows
```
### 3️⃣ Install dependencies 
```bash  
pip install -r requirements.txt 
```
### 4️⃣ Run the chatbot 
```bash 
streamlit run bot.py
```

## 📂 Folder Structure
📂 chatbot-faiss-llm \
│── 📜 bot.py   # Main Streamlit application \
│── 📜 retrieval.py   # Retrieval function using FAISS \
│── 📜 llm_utils.py # Functions to interact with the LLM \
│── 📜 graphs.py # Data visualization functions \
│── 📜 imports.py # Centralized module imports \
📂 data # Folder for uploaded PDF documents \
📂 models # Stored FAISS index and LLM embeddings \
│── 📜 requirements.txt # Python dependencies \
│── 📜 README.md # Project documentation\


## 📝 Usage
✅ Upload one or more PDF documents.\
✅ Ask questions related to the documents.\
✅ The chatbot retrieves relevant content and generates an answer.\
