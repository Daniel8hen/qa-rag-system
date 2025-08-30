# 🤖 QA-RAG-System: A Scalable Question-Answering System with Web & Document Processing
> **Supercharge your document-based question answering using OpenAI's LLMs, ChromaDB, and web scraping!**  

## 🚀 About  
**QA-RAG-System** is a robust and scalable pipeline designed to perform efficient question answering over large collections of documents from multiple sources.  
Using **OpenAI embeddings**, **ChromaDB**, **LangChain**, and **advanced web scraping**, this system can process PDFs, web pages, and mixed document sources to create comprehensive knowledge bases.

Whether you're building a knowledge assistant, research aid, or document search tool, **QA-RAG-System** empowers you to go from raw text and web content to insights in minutes!

---

## 🛠 Features  
- **📄 Multi-Source Processing**: Process PDFs, web pages, and mixed document collections
- **🌐 Web Scraping**: Advanced content extraction from URLs using Trafilatura and BeautifulSoup
- **⚡ Async Batch Processing**: Concurrent document processing with configurable limits  
- **🔄 Content Deduplication**: Automatic duplicate detection using content hashing
- **📊 Rich Metadata**: Source tracking, timestamps, and content attribution
- **🚀 Embedding Generation**: OpenAI embeddings for high-quality document vectors  
- **🧩 Intelligent Chunking**: Smart text splitting optimized for retrieval
- **💾 Vector Store**: Persistent ChromaDB database with incremental updates
- **🤖 Question-Answering Pipeline**: Seamless retrieval and generation
- **🛠 Extensible & Modular**: Easy to add new sources, databases, or LLMs
- **⌨️ Advanced CLI**: Comprehensive command-line interface with rich configuration  

---

## 📂 Project Structure  
```plaintext
qa-rag-system/
├── src/
│   ├── models/                    # LLM and embedding models
│   │   └── model_generator.py     # OpenAI model initialization
│   ├── utils/                     # Core processing utilities  
│   │   ├── document_loader.py     # Multi-source document loading
│   │   └── text_processer.py      # Text chunking and processing
│   ├── data_interactor.py         # ChromaDB operations
│   └── main.py                    # Enhanced CLI entry point
├── tests/                         # Unit tests
├── README.md                      # Project overview
├── WEB_PROCESSING_GUIDE.md        # Web scraping usage guide
├── requirements.txt               # Dependencies (including web scraping)
├── example_sources.json           # Example sources file
└── .env.example                   # Environment configuration
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

### 📄 **Process Documents**

#### Single PDF Processing
```bash
python main.py --action process --pdf_path path_to_your_pdf
```

#### Single or Multiple Web URLs  
```bash
# Single URL
python main.py --action process-urls --urls "https://docs.python.org/3/tutorial/introduction.html"

# Multiple URLs with concurrency control
python main.py --action process-urls \
  --urls "https://docs.python.org/3/tutorial/introduction.html" \
         "https://realpython.com/python-web-scraping-practical-introduction/" \
  --max_concurrent 3 --chunk_size 3000
```

#### Batch Processing from File
Create a JSON file (e.g., `sources.json`) with your document sources:
```json
[
  "https://docs.python.org/3/tutorial/introduction.html",
  "https://docs.python.org/3/tutorial/controlflow.html", 
  "path/to/local/document.pdf",
  "https://realpython.com/python-web-scraping-practical-introduction/"
]
```

Then process:
```bash
python main.py --action process-batch --sources_file sources.json --max_concurrent 5
```

#### Mixed Sources (Command Line)
```bash
python main.py --action process-batch \
  --sources "path/to/document.pdf" \
           "https://example.com/article" \
           "https://docs.example.com/guide" \
  --chunk_size 3000 --chunk_overlap 50
```

### 🔍 **Query Your Knowledge Base**
```bash
python main.py --action query --question "What is web scraping and how does Python support it?"
```

### ⚙️ **Configuration Options**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max_concurrent` | Maximum concurrent downloads | 5 |
| `--chunk_size` | Text chunk size for processing | 4000 |
| `--chunk_overlap` | Overlap between text chunks | 20 |

### 📋 **Complete CLI Reference**
```bash
# Actions
--action process          # Process single PDF
--action process-urls     # Process web URLs  
--action process-batch    # Process mixed sources
--action query           # Query knowledge base

# Input Sources
--pdf_path PATH          # PDF file path
--urls URL1 URL2         # List of URLs
--sources_file FILE      # JSON file with sources
--sources SRC1 SRC2      # Mixed sources list

# Processing Options  
--max_concurrent N       # Concurrent processing limit
--chunk_size N          # Text chunk size
--chunk_overlap N       # Chunk overlap size

# Query
--question "text"       # Question to ask
```

### 💡 **Example Workflows**

**Research Paper Analysis:**
```bash
# Process research papers and documentation
python main.py --action process-batch --sources \
  "paper1.pdf" "paper2.pdf" \
  "https://arxiv.org/abs/example" \
  "https://docs.framework.com"

# Ask research questions  
python main.py --action query --question "Compare the methodologies used in these papers"
```

**Documentation Knowledge Base:**
```bash  
# Build from documentation websites
python main.py --action process-urls \
  --urls "https://docs.python.org/3/" \
         "https://langchain-ai.github.io/langchain/" \
         "https://docs.openai.com/" \
  --max_concurrent 3

# Query the documentation
python main.py --action query --question "How do I implement async processing in Python?"
```


## 🧪 Testing  

**Unit Tests:**
```bash
pytest tests/
```

**Web Processing Tests:**
```bash
python test_web_processing.py
```

**Quick Functionality Test:**
```bash
# Test with sample URLs  
python main.py --action process-urls --urls "https://httpbin.org/html" --max_concurrent 1
python main.py --action query --question "What content was processed?"
```

## 🚨 Important Notes

- **🌐 Web Scraping**: Always respect website terms of service and robots.txt files  
- **⚡ Rate Limiting**: Use `--max_concurrent` responsibly to avoid overwhelming servers
- **🔧 Content Quality**: Some websites may have anti-scraping measures or provide poor content extraction
- **🔐 SSL Handling**: The system handles SSL certificate issues for testing environments
- **📊 Performance**: Larger concurrent limits increase speed but use more resources

## 🆕 What's New in v2.0

- ✨ **Web Document Processing**: Extract content from any web URL
- 🚀 **Concurrent Processing**: Process multiple documents simultaneously  
- 🔄 **Content Deduplication**: Automatic duplicate detection and removal
- 📊 **Rich Metadata**: Enhanced source tracking and content attribution
- 🛠 **Advanced CLI**: More powerful command-line interface
- 📁 **Batch Processing**: Handle large document collections efficiently

## 📖 Additional Resources

- **[Web Processing Guide](WEB_PROCESSING_GUIDE.md)**: Comprehensive guide to web scraping features
- **[Example Sources](example_sources.json)**: Sample JSON file for batch processing

## 🤝 Contributing
Contributions are welcome!


## 🙏 Acknowledgements
A special thanks to the developers of:

* [LangChain](https://github.com/langchain-ai/langchain) for the powerful RAG framework
* [OpenAI](https://openai.com/) for state-of-the-art language models and embeddings
* [ChromaDB](https://www.trychroma.com/) for efficient vector storage and retrieval
* [Trafilatura](https://github.com/adbar/trafilatura) for excellent web content extraction
* [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing capabilities

---

**Ready to build your scalable knowledge base? Start processing documents and asking questions today! 🚀**



