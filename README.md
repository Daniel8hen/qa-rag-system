# 🤖 QA-RAG-System: A Question-Answering System with Retrieval-Augmented Generation (RAG)  
> **Supercharge your document-based question answering using OpenAI's LLMs & ChromaDB!**  

## 🚀 About  
**QA-RAG-System** is a robust and scalable pipeline designed to perform efficient question answering over large text documents.  
Using **OpenAI embeddings**, **ChromaDB**, and **LangChain**, this system splits documents into retrievable chunks, retrieves relevant sections, and generates human-like answers via a language model.

Whether you're building a knowledge assistant, research aid, or document search tool, **QA-RAG-System** empowers you to go from raw text to insights in minutes!

---

## 🛠 Features  
- **Embedding Generation**: Supports OpenAI embeddings for document vectors.  
- **Document Chunking**: Intelligent splitting of PDFs for better retrieval.  
- **Vector Store**: Persistent database using ChromaDB.  
- **Question-Answering Pipeline**: Combines retrieval and generation seamlessly.  
- **Extensible & Modular**: Easy to add new retrievers, databases, or LLMs.  
- **CLI-Friendly**: Fully parametrized for dynamic execution.  

---

## 📂 Project Structure  
```plaintext
qa-rag-system/
├── src/
│   ├── models/             # Encapsulates LLM logic
│   ├── utils/              # Utility scripts
│   ├── main.py             # CLI entry point
├── tests/                  # Unit tests
├── README.md               # Project overview
├── requirements.txt        # Dependencies
├── .env.example            # Example environment file
```

## 🔧 Installation  

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/Daniel8hen/qa-rag-system.git
   cd qa-rag-system
   ```
2. **Set up a virtual environment**:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:

Create a .env file based on .env.example.
Set your OpenAI API key in the .env file, e.g.:
```bash
OPENAI_API_KEY=your_openai_api_key
```

## 🚀 Usage  

1. **Process a new PDF document**:  

To process a PDF and build a ChromaDB index:
```bash
python main.py --action process --pdf_path path_to_your_pdf
```

2. **Ask questions from the indexed data**:

To query the system after the database has been built:
```bash
python main.py --action query --question "Your question here"
```

3. **CLI arguments**:
```bash
--action: Specify the action to perform (process or query).
--pdf_path: Provide the path to the PDF (required for process).
--question: Provide the question to ask (required for query).
```

4. **Example Commands**:
To process a sample PDF:
```bash
python main.py --action process --pdf_path datascience_paper.pdf
```

To ask a question:
```bash
python main.py --action query --question "What are the key highlights of the paper?"
```


## 🧪 Testing  

To ensure the functionality works as expected, you can run the tests:  
```bash
pytest tests/
```
Tests are located in the tests directory, covering core modules and edge cases.

## 🤝 Contributing
Contributions are welcome!


## 🙏 Acknowledgements
A special thanks to the developers of:

* [LangChain](https://github.com/langchain-ai/langchain) for the powerful framework
* [OpenAI](https://openai.com/) for the state-of-the-art language models
* [ChromaDB](https://www.trychroma.com/) for efficient vector storage and retrieval



