# LLM-Powered Intelligent Query-Retrieval System

An AI system that processes large documents (PDF, DOCX) and responds to natural language queries with contextual, clause-based decisions using GPT-4 and vector similarity search.

## 🚀 Features

- **Document Processing**: Ingests PDF and DOCX files from blob URLs
- **Natural Language Queries**: Processes complex questions about document content
- **Vector Search**: Uses Pinecone or in-memory FAISS for semantic similarity
- **LLM Reasoning**: GPT-4 powered logic evaluation and answer generation
- **Fast Response**: Optimized for <30 second response times
- **Structured Output**: Returns clean JSON responses with explanations

## 🏗️ Architecture

The system consists of 6 main components:

1. **Document Processor**: Downloads and parses PDF/DOCX files
2. **LLM Service**: GPT-4 integration for query parsing and reasoning
3. **Vector Service**: Pinecone/FAISS for embedding storage and search
4. **Query Processor**: Main orchestrator for processing queries
5. **API Layer**: FastAPI with authentication and validation
6. **Response Formatter**: Structured JSON output generation

## 📋 Requirements

- Python 3.11+
- OpenAI API key (GPT-4 access)
- Pinecone API key (optional, falls back to in-memory search)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd llm-query-retrieval-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=document-embeddings
```

## 🚀 Local Development

1. **Start the server**
```bash
python main.py
# or
uvicorn main:app --reload
```

2. **Test the API**
```bash
python test_api.py
```

3. **Access documentation**
- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🌐 Deployment

### Vercel (Recommended)

1. **Install Vercel CLI**
```bash
npm i -g vercel
```

2. **Deploy**
```bash
vercel --prod
```

3. **Set environment variables in Vercel dashboard**

### Railway

1. **Connect GitHub repository to Railway**
2. **Set environment variables**
3. **Deploy automatically on push**

### Heroku

1. **Create Heroku app**
```bash
heroku create your-app-name
```

2. **Set environment variables**
```bash
heroku config:set OPENAI_API_KEY=your_key
heroku config:set PINECONE_API_KEY=your_key
```

3. **Deploy**
```bash
git push heroku main
```

### Docker

1. **Build image**
```bash
docker build -t llm-query-system .
```

2. **Run container**
```bash
docker run -p 8000:8000 --env-file .env llm-query-system
```

## 📡 API Usage

### Authentication
All endpoints require Bearer token authentication:
```
Authorization: Bearer 12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f
```

### Main Endpoint

**POST** `/api/v1/hackrx/run`

**Request:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What are the coverage conditions?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The grace period for premium payment is 30 days from the due date.",
    "Coverage conditions include a 2-year waiting period for pre-existing diseases."
  ]
}
```

### Other Endpoints

- **GET** `/` - Health check
- **GET** `/health` - Detailed health status
- **POST** `/api/v1/preprocess` - Preprocess document for faster queries
- **DELETE** `/api/v1/cache` - Clear document cache

## 🧪 Testing

Run the test suite:
```bash
python test_api.py
```

For deployed API:
```bash
API_URL=https://your-deployed-url.com python test_api.py
```

## 🔧 Configuration

### Performance Tuning

- **Chunk Size**: Adjust `chunk_size` in `DocumentProcessor` (default: 1000 chars)
- **Vector Search**: Modify `top_k` parameter for more/fewer results
- **Token Limits**: Configure `max_tokens` in `LLMService`

### Caching

The system includes document caching to avoid reprocessing:
- Documents are cached after first processing
- Use `/api/v1/cache` endpoint to clear cache
- Cache persists for the application lifetime

## 📊 Monitoring

The API includes logging and timing information:
- Request processing times
- Error tracking
- Document processing statistics

## 🔒 Security

- Bearer token authentication required
- HTTPS enforced in production
- Input validation and sanitization
- Rate limiting recommended for production

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Check API key validity
   - Verify GPT-4 access
   - Monitor rate limits

2. **Pinecone Connection Issues**
   - Verify API key and environment
   - Check index name configuration
   - System falls back to in-memory search

3. **Document Processing Errors**
   - Ensure document URL is accessible
   - Check file format (PDF/DOCX supported)
   - Verify blob URL permissions

4. **Slow Response Times**
   - Use document preprocessing endpoint
   - Reduce number of questions per request
   - Optimize chunk size and search parameters

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request
