# 🏥 Build a Complete Medical Chatbot with LLMs, LangChain, Pinecone, Flask & AWS

An end-to-end Medical Chatbot built using **LangChain, Groq LLM, HuggingFace Embeddings, Pinecone Vector Database, and Flask**.

---

# 🚀 How to Run the Project

## 📌 STEP 0 – Clone the Repository

```bash
git clone https://github.com/gowtham-dd/GenAiBOTEndToEnd
cd GenAiBOTEndToEnd
```

---

## 📌 STEP 1 – Create Virtual Environment (Python 3.11)

Make sure Python 3.11 is installed:

```bash
python --version
```

Create a virtual environment:

```bash
python3.11 -m venv medibot
```

Activate the environment:

### ▶ Windows
```bash
medibot\Scripts\activate
```

### ▶ Mac/Linux
```bash
source medibot/bin/activate
```

---

## 📌 STEP 2 – Install Dependencies

Upgrade pip and install requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📌 STEP 3 – Setup Environment Variables

Create a `.env` file in the root directory and add:

```ini
PINECONE_API_KEY="your_pinecone_api_key"
GROQ_API_KEY="your_groq_api_key"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_token"
```

Make sure:
- No extra spaces before variable names  
- API keys are valid  
- `.env` file is placed in the root directory  

---

## 📌 STEP 4 – Create Pinecone Index

Create a Pinecone index with:

- **Index Name:** `medical-chatbot`  
- **Dimension:** `384`  
- **Metric:** `cosine`  

---

## 📌 STEP 5 – Create Namespace

Create a namespace in Pinecone named:

```
medical-bot
```

---

## 📌 STEP 6 – Store Embeddings

Run the following command to store embeddings into Pinecone:

```bash
python store_index.py
```

---

## 📌 STEP 7 – Run the Application

```bash
python app.py
```

---

## 🌐 Access the Application

Open your browser and go to:

```
http://localhost:8080
```

(or the port specified in `app.py`)

---

# 🛠 Tech Stack Used

- Python 3.11  
- LangChain  
- Flask  
- Groq LLM  
- HuggingFace Embeddings  
- Pinecone Vector Database  

---

# 📦 Project Workflow

1. Load medical documents  
2. Convert text → embeddings (384-dimension vectors)  
3. Store embeddings inside Pinecone (`medical-bot` namespace)  
4. Perform similarity search  
5. Send retrieved context to Groq LLM  
6. Generate medical response  

---

## 👨‍💻 Author

Gowtham D  

---

⭐ If you found this helpful, consider giving the repository a star!