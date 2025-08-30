# Web Document Processing Guide

Your QA-RAG system now supports scalable web document processing! Here's how to use the new features:

## 🌐 New Capabilities

- **Web Scraping**: Extract content from URLs using advanced content extraction
- **Batch Processing**: Process multiple URLs concurrently for better performance  
- **Mixed Sources**: Combine PDFs and web documents in a single knowledge base
- **Deduplication**: Automatically remove duplicate content based on content hashing
- **Async Processing**: Concurrent document loading with configurable limits

## 📖 Usage Examples

### Process Single URL
```bash
python main.py --action process-urls --urls "https://docs.python.org/3/tutorial/introduction.html"
```

### Process Multiple URLs
```bash
python main.py --action process-urls \
  --urls "https://docs.python.org/3/tutorial/introduction.html" \
         "https://realpython.com/python-web-scraping-practical-introduction/" \
  --max_concurrent 3
```

### Batch Process from File
Create a JSON file with your sources:
```json
[
  "https://docs.python.org/3/tutorial/introduction.html",
  "https://docs.python.org/3/tutorial/controlflow.html",
  "https://realpython.com/python-web-scraping-practical-introduction/"
]
```

Then process:
```bash
python main.py --action process-batch --sources_file sources.json --max_concurrent 5
```

### Mix URLs and PDFs
```bash
python main.py --action process-batch \
  --sources "path/to/document.pdf" \
           "https://example.com/article" \
           "https://docs.example.com/guide" \
  --chunk_size 3000 --chunk_overlap 50
```

### Query Your Web-Processed Knowledge Base
```bash
python main.py --action query --question "What is web scraping and how does Python support it?"
```

## ⚙️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max_concurrent` | Maximum concurrent downloads | 5 |
| `--chunk_size` | Text chunk size for processing | 4000 |
| `--chunk_overlap` | Overlap between text chunks | 20 |

## 🛠 Advanced Features

### Content Extraction
- Uses **Trafilatura** for high-quality content extraction from web pages
- Falls back to **BeautifulSoup** for pages that Trafilatura cannot process
- Filters out pages with very little meaningful content

### Error Handling
- Gracefully handles HTTP errors, SSL issues, and malformed content
- Logs failed URLs while continuing to process successful ones
- Validates extracted content before adding to knowledge base

### Performance Optimizations
- Async HTTP requests with configurable concurrency limits
- Content deduplication using MD5 hashing
- Metadata tracking for source attribution and debugging

## 📊 Metadata Tracking

Each processed document includes rich metadata:
- `source_type`: "web" or "pdf"
- `source_url`: Original URL for web documents
- `content_hash`: MD5 hash for deduplication
- `processed_at`: Processing timestamp
- `title`: Extracted page title
- `content_length`: Character count

## 🚨 Important Notes

- **Rate Limiting**: Use `--max_concurrent` responsibly to avoid overwhelming servers
- **Content Quality**: Some websites may have anti-scraping measures or poor content extraction
- **SSL Issues**: The system handles SSL certificate issues for testing, but consider proper certificates for production
- **Terms of Service**: Always respect website terms of service and robots.txt files

## 🧪 Testing

Run the test suite to verify functionality:
```bash
python test_web_processing.py
```

## 🔧 Dependencies Added
- `beautifulsoup4==4.12.3`: HTML parsing
- `trafilatura==1.12.2`: Content extraction  
- `lxml==5.3.0`: XML/HTML processing

Your system now scales from single PDFs to processing dozens of web documents concurrently!