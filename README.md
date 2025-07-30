# 🧠 LLM-Powered Intelligent Query-Retrieval System

**HackRX 2025 Submission** - An advanced AI system that processes large documents (PDF, DOCX) and responds to natural language queries with contextual, clause-based decisions using Google Gemini AI and vector similarity search.

## 🌐 **Live Demo**
**🚀 Try it now:** [https://llm-query-retrieval-system-production.up.railway.app/](https://llm-query-retrieval-system-production.up.railway.app/)

## 🎯 **HackRX 2025 Features**

- **🔍 Intelligent Document Analysis**: Processes PDF and DOCX files from URLs
- **💬 Natural Language Queries**: Ask complex questions about document content
- **🧠 AI-Powered Reasoning**: Google Gemini AI for advanced logic evaluation
- **⚡ Fast Response**: Optimized for <30 second response times
- **🎨 Modern Web Interface**: Professional, responsive frontend design
- **📊 Structured Output**: Clean JSON responses with detailed explanations
- **🔒 Secure API**: Bearer token authentication and HTTPS encryption

## 🏗️ **System Architecture**

**Modern, scalable architecture with 6 core components:**

1. **📄 Document Processor**: Downloads and parses PDF/DOCX files with advanced text extraction
2. **🤖 LLM Service**: Google Gemini AI integration for query parsing and intelligent reasoning
3. **🔍 Vector Service**: Pinecone/FAISS for semantic similarity search and embeddings
4. **⚙️ Query Processor**: Main orchestrator coordinating all processing components
5. **🌐 API Layer**: FastAPI with authentication, validation, and comprehensive error handling
6. **📊 Response Formatter**: Structured JSON output with detailed explanations

## 📋 **Requirements**

- **Python 3.11+**
- **Google Gemini API key** (AI reasoning and processing)
- **Pinecone API key** (optional, falls back to in-memory search)
- **Railway/Vercel account** (for deployment)

## 🌟 **HackRX 2025 Submission URLs**

| Service | URL | Purpose |
|---------|-----|---------|
| **🌐 Live Demo** | [https://llm-query-retrieval-system-production.up.railway.app/](https://llm-query-retrieval-system-production.up.railway.app/) | Interactive web interface |
| **🔌 API Endpoint** | `https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run` | Main processing API (POST) |
| **📚 API Docs** | [https://llm-query-retrieval-system-production.up.railway.app/docs](https://llm-query-retrieval-system-production.up.railway.app/docs) | Interactive API documentation |
| **📡 Webhook** | `https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/status` | HackRX daily updates (GET) |

## 🛠️ **Quick Start**

### **Option 1: Use Live Demo (Recommended)**
Just visit: [https://llm-query-retrieval-system-production.up.railway.app/](https://llm-query-retrieval-system-production.up.railway.app/)

### **Option 2: Local Development**

1. **Clone the repository**
```bash
git clone https://github.com/MukundC25/llm-query-retrieval-system.git
cd llm-query-retrieval-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file with:
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp-free
PINECONE_INDEX_NAME=document-embeddings
BEARER_TOKEN=your_bearer_token_here
```

4. **Start the server**
```bash
python run.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

5. **Access the application**
- **Frontend**: http://localhost:8000/
- **API docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

## 🌐 **Deployment (Railway)**

**Current deployment is live on Railway:**

1. **Automatic deployment** from GitHub repository
2. **Environment variables** configured in Railway dashboard:
   - `GEMINI_API_KEY=your_gemini_api_key`
   - `PINECONE_API_KEY=your_pinecone_key`
   - `PINECONE_ENVIRONMENT=us-west1-gcp-free`
   - `BEARER_TOKEN=your_bearer_token`

3. **Custom domain**: https://llm-query-retrieval-system-production.up.railway.app/

### **Deploy Your Own Instance**

**Railway (Recommended):**
1. Fork this repository
2. Connect to Railway
3. Set environment variables
4. Deploy automatically

**Vercel:**
```bash
npm i -g vercel
vercel --prod
```

**Docker:**
```bash
docker build -t llm-query-system .
docker run -p 8000:8000 --env-file .env llm-query-system
```

## 📡 **API Usage**

### **🔒 Authentication**
All API endpoints require Bearer token authentication:
```bash
Authorization: Bearer YOUR_BEARER_TOKEN
```

### **🎯 Main Endpoint**

**POST** `/api/v1/hackrx/run`

**Example Request:**
```bash
curl -X POST "https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run" \
  -H "Authorization: Bearer 12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What are the coverage conditions?"
    ]
  }'
```

**Response:**
```json
{
  "answers": [
    "The grace period for premium payment is 30 days from the due date as specified in Section 4.2 of the policy document.",
    "Coverage conditions include a 2-year waiting period for pre-existing diseases and immediate coverage for accidents as outlined in Section 3.1."
  ]
}
```

### **🔧 Additional Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| **GET** | `/` | Frontend web interface |
| **GET** | `/health` | API health status |
| **GET** | `/docs` | Interactive API documentation |
| **GET** | `/api/v1/hackrx/status` | HackRX project status |

## 🧪 **Testing**

### **Quick Test (Live API)**
```bash
curl -X POST "https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run" \
  -H "Authorization: Bearer YOUR_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf", "questions": ["What is this document about?"]}'
```

### **Local Testing**
```bash
python test_api.py
```

## ⚙️ **Technical Specifications**

### **🚀 Performance**
- **Response Time**: <30 seconds for most documents
- **Document Size**: Up to 50MB supported
- **Concurrent Requests**: Optimized for multiple simultaneous queries
- **Timeout Handling**: 25-second server timeout with graceful error handling

### **🔧 Configuration**
- **Chunk Size**: 1000 characters with 200 character overlap
- **Vector Search**: Top-5 semantic similarity results
- **LLM Model**: Google Gemini Pro with 0.1 temperature
- **Max Tokens**: 4000 tokens per response

### **💾 Caching & Storage**
- **Document Caching**: Automatic caching for faster repeated queries
- **Vector Storage**: Pinecone cloud or in-memory FAISS fallback
- **Session Management**: Stateless API design

## 🔒 **Security & Compliance**

- **🔐 Authentication**: Bearer token required for all endpoints
- **🌐 HTTPS**: TLS encryption enforced in production
- **✅ Input Validation**: Comprehensive request validation and sanitization
- **🛡️ Error Handling**: Secure error responses without sensitive data exposure
- **📝 Logging**: Comprehensive audit trail for all requests

## 🐛 **Troubleshooting**

### **Common Issues & Solutions**

| Issue | Cause | Solution |
|-------|-------|----------|
| **🔴 502 Error** | Server timeout or overload | Try smaller document or fewer questions |
| **🔑 Authentication Failed** | Invalid bearer token | Check your bearer token in environment variables |
| **📄 Document Not Found** | Invalid URL or permissions | Ensure document URL is publicly accessible |
| **⏱️ Slow Response** | Large document processing | Use shorter documents or preprocess endpoint |
| **🤖 Gemini API Error** | API key or rate limits | Verify your Gemini API key configuration |

### **🆘 Support**
- **📚 Documentation**: [API Docs](https://llm-query-retrieval-system-production.up.railway.app/docs)
- **🔍 Health Check**: [System Status](https://llm-query-retrieval-system-production.up.railway.app/health)
- **💬 Issues**: [GitHub Issues](https://github.com/MukundC25/llm-query-retrieval-system/issues)

## 🏆 **HackRX 2025 Submission**

**Project**: LLM-Powered Intelligent Query-Retrieval System
**Team**: AI Innovators
**Tech Stack**: FastAPI + Google Gemini + Railway + Tailwind CSS
**Demo**: [https://llm-query-retrieval-system-production.up.railway.app/](https://llm-query-retrieval-system-production.up.railway.app/)

### **🎯 Submission Checklist**
- ✅ **Live Demo URL**: Working web interface
- ✅ **API Endpoint**: Functional REST API
- ✅ **Webhook URL**: Daily updates endpoint
- ✅ **Documentation**: Comprehensive API docs
- ✅ **Authentication**: Secure bearer token system
- ✅ **Performance**: <30 second response times
- ✅ **Error Handling**: Robust error management
- ✅ **Frontend**: Professional web interface

## 📝 **License**

MIT License - Feel free to use and modify for your projects.

## 🤝 **Contributing**

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

**⭐ Star this repository if you found it helpful!**

**🚀 Built for HackRX 2025 with ❤️**
