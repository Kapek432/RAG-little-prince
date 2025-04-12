# 🧠 RAG with "The Little Prince" 📚

A simple Retrieval-Augmented Generation (RAG) pipeline using **LangChain**, **Ollama**, and **ChromaDB** on *The Little Prince*.

## 📌 Features

- Loads and chunks PDFs using LangChain
- Stores chunks in Chroma vector DB
- Embeds using `nomic-embed-text` via Ollama
- Answers questions using `mistral` model with relevant context

---

## ⚠️ Disclaimer: Showcase Model

Please note that this model is intended as a **showcase** and **proof of concept**. The responses may not be of the highest quality, as the models used are relatively smaller in scale compared to what might be needed for production-level tasks. 

For **better results** and more robust answers, I recommend using advanced models, such as:

- **OpenAI's GPT models** for highly accurate and contextual responses.
- **AWS Bedrock** for scalable, enterprise-level AI solutions.

---

## 📂 Project Structure

- `books/`: Folder containing the `the_little_prince.pdf`
- `database.py`: Loads and splits the document into vector DB
- `query.py`: Queries your PDF using Mistral + context
- `get_embedding_function.py`: Wraps embedding model
- `chroma/`: Persistent vector store (auto-generated)
- `requirements.txt`: Required Python packages
- `README.md`: Documentation for the project

---

## 🛠️ Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/Kapek432/rag-little-prince.git
cd rag-little-prince
```
Here’s the updated `README.md` with a note encouraging users to try with other books, while recommending not using large ones for better performance.

---



### Step 2: Set up a virtual environment (Optional but recommended)

It's recommended to create a virtual environment to keep your dependencies isolated.

1. **Create the environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the environment**:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

### Step 3: Install dependencies

Once the virtual environment is activated, install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 📥 Ollama Setup (Required)

This project uses [**Ollama**](https://ollama.com/) to run both the embedding and LLM models locally. Please follow the steps below to set it up.

### Step 1: Install Ollama

- Go to [**Ollama Download**](https://ollama.com/download) and follow the instructions to install Ollama for your operating system (Windows, macOS, or Linux).

### Step 2: Download required models

Once Ollama is installed, you need to download the following models for the RAG pipeline:

1. **Download the embedding model**:
   ```bash
   ollama run nomic-embed-text
   ```

2. **Download the Mistral model**:
   ```bash
   ollama run mistral
   ```

These models are required to generate embeddings and responses for the query.

---

## 📥 Ingest the Document

Before querying, you need to ingest *The Little Prince* PDF into the Chroma database. 

### Step 1: Place your document

1. **Download the PDF** of *The Little Prince* from [Project Gutenberg](https://www.gutenberg.org/ebooks/21603), or use your own copy.
2. **Place the `the_little_prince.pdf` file** in the `books/` folder within the project directory.

### Step 2: Ingest the document into Chroma DB

Run the following command to process the PDF, chunk it into smaller parts, and store it in the Chroma vector database. 

To reset the database (recommended for the first time):

```bash
python database.py --reset
```

If you don’t need to reset and just want to add or update the existing data:

```bash
python database.py
```

---

## ❓ Ask Questions

Once the document is ingested and indexed, you can query the system to retrieve information based on *The Little Prince*.

### Step 1: Ask a question

Run the query script to ask any question based on the content from the PDF. For example:

```bash
python query.py "What is the main message of the book?"
```

This will retrieve the most relevant chunks from the document and generate a response using the **Mistral** model with context from *The Little Prince*.

### Example Output (in my case)

```
Response:  The main message of the book appears to be the importance of taking responsibility for one's actions and the things that one cares about, as well as understanding life and its complexities. It also emphasizes the value of friendships and empathy, even in difficult or dangerous situations. Additionally, it suggests that holding onto memories can help us remember and appreciate those who are no longer with us.
Sources: ['books\\saint_exupery_antoine-the_little_prince.pdf:54:0', 'books\\saint_exupery_antoine-the_little_prince.pdf:8:0', 'books\\saint_exupery_antoine-the_little_prince.pdf:41:1', 'books\\saint_exupery_antoine-the_little_prince.pdf:31:0', 'books\\saint_exupery_antoine-the_little_prince.pdf:44:0']
```
---

## 📚 Try with Other Books

You may try this pipeline with other books by simply replacing *The Little Prince* PDF with your chosen book's PDF file. However, **it's recommended not to use a very large book** as this may impact performance. Smaller books work best for quicker processing and better response times.

---

## 📄 License

MIT — free to use and modify. Credit to Antoine de Saint-Exupéry for the timeless story.


---

