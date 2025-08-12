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
| **📡 Webhook** | `https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run` | HackRX daily updates (GET) |

---

## 🛠️ **⚡ Quick Start Guide ⚡**

<div align="center">

### 🌟 **Option 1: Live Demo (Recommended)** 🌟

<a href="https://llm-query-retrieval-system-production.up.railway.app/">
<img src="https://img.shields.io/badge/🚀_Click_Here_to_Try_Live_Demo!-brightgreen?style=for-the-badge&logo=rocket&logoColor=white" alt="Live Demo" />
</a>

**✨ No setup required! Just click and start querying documents! ✨**

</div>

---

### 🔧 **Option 2: Local Development**

<table>
<tr>
<td width="50%">

#### 📥 **Step 1: Clone & Setup**
```bash
# 🔄 Clone the repository
git clone https://github.com/MukundC25/llm-query-retrieval-system.git
cd llm-query-retrieval-system

# 📦 Install dependencies
pip install -r requirements.txt
```

#### 🔑 **Step 2: Environment Setup**
```bash
# 📝 Create .env file with:
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp-free
PINECONE_INDEX_NAME=document-embeddings
BEARER_TOKEN=your_bearer_token_here
```

</td>
<td width="50%">

#### 🚀 **Step 3: Launch Server**
```bash
# 🎯 Option A: Simple start
python run.py

# 🔧 Option B: Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 🌐 **Step 4: Access Application**
- 🎨 **Frontend**: http://localhost:8000/
- 📚 **API docs**: http://localhost:8000/docs
- ❤️ **Health check**: http://localhost:8000/health

<div align="center">
<img src="https://img.shields.io/badge/✅_Ready_to_Go!-success?style=for-the-badge&logo=checkmark" alt="Ready" />
</div>

</td>
</tr>
</table>

---

## 🌐 **🚀 Deployment Options 🚀**

<div align="center">

### 🎯 **Current Live Deployment**

<img src="https://img.shields.io/badge/🚂_Railway-Live_&_Running-brightgreen?style=for-the-badge&logo=railway" alt="Railway Status" />

**✨ Automatic deployment from GitHub repository ✨**

</div>

<table>
<tr>
<td width="50%">

### 🔧 **Environment Configuration**
```bash
# 🔑 Required API Keys
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-west1-gcp-free
PINECONE_INDEX_NAME=document-embeddings
BEARER_TOKEN=your_bearer_token
```

### 🌍 **Live Domain**
🔗 **Production URL:**
https://llm-query-retrieval-system-production.up.railway.app/

</td>
<td width="50%">

### 🚀 **Deploy Your Own Instance**

#### 🚂 **Railway (Recommended)**
1. 🍴 Fork this repository
2. 🔗 Connect to Railway
3. ⚙️ Set environment variables
4. 🚀 Deploy automatically

#### ▲ **Vercel**
```bash
npm i -g vercel
vercel --prod
```

#### 🐳 **Docker**
```bash
docker build -t llm-query-system .
docker run -p 8000:8000 --env-file .env llm-query-system
```

</td>
</tr>
</table>

---

## 📡 **🔌 API Usage Guide 🔌**

<div align="center">

### 🔒 **Authentication Required**

<img src="https://img.shields.io/badge/🔑_Bearer_Token-Required-red?style=for-the-badge&logo=shield" alt="Auth Required" />

```bash
Authorization: Bearer YOUR_BEARER_TOKEN
```

</div>

---

### 🎯 **Main Processing Endpoint**

<table>
<tr>
<td width="50%">

#### 📤 **Request Example**
```bash
curl -X POST "https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What are the coverage conditions?"
    ]
  }'
```

</td>
<td width="50%">

#### 📥 **Response Example**
```json
{
  "answers": [
    "The grace period for premium payment is 30 days from the due date as specified in Section 4.2 of the policy document.",
    "Coverage conditions include a 2-year waiting period for pre-existing diseases and immediate coverage for accidents as outlined in Section 3.1."
  ]
}
```

<div align="center">
<img src="https://img.shields.io/badge/✨_JSON_Response-Structured-blue?style=for-the-badge&logo=json" alt="JSON Response" />
</div>

</td>
</tr>
</table>

---

### 🔧 **Additional Endpoints**

<div align="center">

| 🎯 **Method** | 🔗 **Endpoint** | 📝 **Description** |
|---------------|-----------------|---------------------|
| **GET** 🌐 | `/` | Frontend web interface |
| **GET** ❤️ | `/health` | API health status |
| **GET** 📚 | `/docs` | Interactive API documentation |
| **GET** 📊 | `/api/v1/hackrx/status` | HackRX project status |

</div>

---

## 🧪 **⚡ Testing & Validation ⚡**

<div align="center">

### 🚀 **Quick Live API Test**

<img src="https://img.shields.io/badge/🧪_Test_Live_API-Click_to_Copy-yellow?style=for-the-badge&logo=test-tube" alt="Test API" />

</div>

```bash
curl -X POST "https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run" \
  -H "Authorization: Bearer YOUR_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf", "questions": ["What is this document about?"]}'
```

<div align="center">

### 🔧 **Local Testing**

```bash
python test_api.py
```

<img src="https://img.shields.io/badge/✅_All_Tests_Passing-green?style=for-the-badge&logo=checkmark" alt="Tests Passing" />

</div>

---

## ⚙️ **🔧 Technical Specifications 🔧**

<table>
<tr>
<td width="33%">

### 🚀 **Performance Metrics**
- ⚡ **Response Time**: <30 seconds
- 📄 **Document Size**: Up to 50MB
- 🔄 **Concurrent Requests**: Multi-query support
- ⏱️ **Timeout**: 25s with graceful handling

</td>
<td width="33%">

### 🔧 **AI Configuration**
- 📊 **Chunk Size**: 1000 chars + 200 overlap
- 🔍 **Vector Search**: Top-5 similarity results
- 🤖 **LLM Model**: Google Gemini Pro
- 🎯 **Temperature**: 0.1 for precision
- 📝 **Max Tokens**: 4000 per response

</td>
<td width="33%">

### 💾 **Storage & Caching**
- 🔄 **Document Caching**: Auto-enabled
- 📌 **Vector Storage**: Pinecone/FAISS
- 🌐 **Session Management**: Stateless design
- ⚡ **Memory**: Optimized for speed

</td>
</tr>
</table>

---

## 🔒 **🛡️ Security & Compliance 🛡️**

<div align="center">

<img src="https://img.shields.io/badge/🔐_Security-Enterprise_Grade-green?style=for-the-badge&logo=shield" alt="Security" />
<img src="https://img.shields.io/badge/🌐_HTTPS-TLS_Encrypted-blue?style=for-the-badge&logo=lock" alt="HTTPS" />
<img src="https://img.shields.io/badge/✅_Validated-Input_Sanitized-orange?style=for-the-badge&logo=checkmark" alt="Validated" />

</div>

<table>
<tr>
<td width="50%">

### 🔐 **Authentication & Access**
- 🔑 **Bearer Token**: Required for all endpoints
- 🌐 **HTTPS**: TLS encryption enforced
- ✅ **Input Validation**: Comprehensive sanitization
- 🛡️ **Error Handling**: Secure responses

</td>
<td width="50%">

### 📝 **Monitoring & Compliance**
- 📊 **Audit Trail**: Complete request logging
- 🔍 **Health Monitoring**: Real-time status
- 🚨 **Error Tracking**: Detailed diagnostics
- 🔒 **Data Privacy**: No sensitive data exposure

</td>
</tr>
</table>

---

## 🐛 **🔧 Troubleshooting Guide 🔧**

<div align="center">

### 🚨 **Common Issues & Quick Fixes** 🚨

</div>

<table>
<tr>
<th>🔴 Issue</th>
<th>🔍 Cause</th>
<th>✅ Solution</th>
</tr>
<tr>
<td><strong>🔴 502 Error</strong></td>
<td>Server timeout or overload</td>
<td>Try smaller document or fewer questions</td>
</tr>
<tr>
<td><strong>🔑 Authentication Failed</strong></td>
<td>Invalid bearer token</td>
<td>Check your bearer token in environment variables</td>
</tr>
<tr>
<td><strong>📄 Document Not Found</strong></td>
<td>Invalid URL or permissions</td>
<td>Ensure document URL is publicly accessible</td>
</tr>
<tr>
<td><strong>⏱️ Slow Response</strong></td>
<td>Large document processing</td>
<td>Use shorter documents or preprocess endpoint</td>
</tr>
<tr>
<td><strong>🤖 Gemini API Error</strong></td>
<td>API key or rate limits</td>
<td>Verify your Gemini API key configuration</td>
</tr>
</table>

---

### 🆘 **Support & Resources**

<div align="center">

[![📚 Documentation](https://img.shields.io/badge/📚_Documentation-API_Docs-blue?style=for-the-badge&logo=book)](https://llm-query-retrieval-system-production.up.railway.app/docs)
[![🔍 Health Check](https://img.shields.io/badge/🔍_Health_Check-System_Status-green?style=for-the-badge&logo=heart)](https://llm-query-retrieval-system-production.up.railway.app/health)
[![💬 Issues](https://img.shields.io/badge/💬_Issues-GitHub_Support-orange?style=for-the-badge&logo=github)](https://github.com/MukundC25/llm-query-retrieval-system/issues)

</div>

---

## 🏆 **🎉 HackRX 2025 Submission 🎉**

<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=2000&pause=1000&color=FF6B35&center=true&vCenter=true&width=500&lines=🏆+HackRX+2025+Official+Submission+🏆;🤖+AI+Innovators+Team+🤖;⚡+FastAPI+%2B+Google+Gemini+⚡;🚀+Deployed+on+Railway+🚀" alt="HackRX Typing SVG" />

<br/>

**🎯 Project**: LLM-Powered Intelligent Query-Retrieval System
**👥 Team**: The Semicolons
**🛠️ Tech Stack**: FastAPI + Google Gemini + Railway + Tailwind CSS
**🌐 Demo**: [Live Application](https://llm-query-retrieval-system-production.up.railway.app/)

</div>

---

### 🎯 **Submission Checklist**

<div align="center">

| Feature | Status | Description |
|---------|--------|-------------|
| 🌐 **Live Demo URL** | ✅ | Working web interface |
| 🔌 **API Endpoint** | ✅ | Functional REST API |
| 📡 **Webhook URL** | ✅ | Daily updates endpoint |
| 📚 **Documentation** | ✅ | Comprehensive API docs |
| 🔒 **Authentication** | ✅ | Secure bearer token system |
| ⚡ **Performance** | ✅ | <30 second response times |
| 🛡️ **Error Handling** | ✅ | Robust error management |
| 🎨 **Frontend** | ✅ | Professional web interface |

</div>

---

## 📝 **📄 License 📄**

<div align="center">

<img src="https://img.shields.io/badge/📄_License-MIT-green?style=for-the-badge&logo=opensourceinitiative" alt="MIT License" />

**MIT License - Feel free to use and modify for your projects! 🚀**

</div>

---

## 🤝 **🌟 Contributing 🌟**

<div align="center">

**We welcome contributions! Here's how to get started:**

</div>

<table>
<tr>
<td width="50%">

### 🔄 **Quick Steps**
1. 🍴 **Fork** the repository
2. 🌿 **Create** a feature branch
   `git checkout -b feature/amazing-feature`
3. 💾 **Commit** your changes
   `git commit -m 'Add amazing feature'`

</td>
<td width="50%">

### 🚀 **Submit Changes**
4. 📤 **Push** to the branch
   `git push origin feature/amazing-feature`
5. 🔄 **Open** a Pull Request
6. 🎉 **Celebrate** your contribution!

</td>
</tr>
</table>

---

<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=FFD700&center=true&vCenter=true&width=600&lines=⭐+Star+this+repository+if+helpful!+⭐;🚀+Built+for+HackRX+2025+with+❤️+🚀;🌟+Thank+you+for+visiting!+🌟" alt="Footer Typing SVG" />

<br/>

[![⭐ Star Repository](https://img.shields.io/badge/⭐_Star_Repository-If_Helpful!-yellow?style=for-the-badge&logo=star)](https://github.com/MukundC25/llm-query-retrieval-system)
[![🚀 HackRX 2025](https://img.shields.io/badge/🚀_HackRX_2025-Built_with_❤️-red?style=for-the-badge&logo=heart)](https://llm-query-retrieval-system-production.up.railway.app/)

<br/>

**🎉 Made with ❤️ by the The Semicolons Team for HackRX 2025 🎉**

</div>
