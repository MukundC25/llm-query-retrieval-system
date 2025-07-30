# 🚀 Deployment Guide - LLM Query Retrieval System

**HackRX 2025 Submission** - Complete deployment guide for the LLM-Powered Intelligent Query-Retrieval System.

## 🌟 **Current Live Deployment**

**✅ Successfully deployed on Railway:**
- **🌐 Live Demo**: [https://llm-query-retrieval-system-production.up.railway.app/](https://llm-query-retrieval-system-production.up.railway.app/)
- **🔌 API Endpoint**: `https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run`
- **📚 Documentation**: [https://llm-query-retrieval-system-production.up.railway.app/docs](https://llm-query-retrieval-system-production.up.railway.app/docs)
- **🔧 Status**: Fully operational with frontend and API

## 🌐 **Platform-Specific Deployment**

### 1. Railway (Currently Used - Recommended)

**✅ Current Production Deployment**

**Prerequisites:**
- Railway account
- GitHub repository (already connected)

**Current Configuration:**
```
Repository: https://github.com/MukundC25/llm-query-retrieval-system
Branch: main
Auto-deploy: Enabled
Domain: llm-query-retrieval-system-production.up.railway.app
```

**Environment Variables (Required):**
```
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp-free
BEARER_TOKEN=your_bearer_token_here
PORT=8000
```

**Configuration Files:**
- `railway.json` - Railway-specific configuration
- `run.py` - Application entry point
- `requirements.txt` - Dependencies

### 2. Vercel (Alternative)

**Prerequisites:**
- Railway account
- GitHub repository

**Steps:**

1. **Connect Repository**
   - Go to Railway dashboard
   - Click "New Project"
   - Connect your GitHub repository

2. **Set Environment Variables**
   In Railway dashboard → Variables:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=us-west1-gcp-free
   PINECONE_INDEX_NAME=document-embeddings
   BEARER_TOKEN=your_bearer_token_here
   PORT=8000
   ```

3. **Deploy**
   - Railway automatically deploys on push to main branch
   - Uses `Procfile` for startup command

### 3. Render

**Prerequisites:**
- Render account
- GitHub repository

**Steps:**

1. **Create Web Service**
   - Go to Render dashboard
   - Click "New" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**
   ```
   Name: llm-query-retrieval-system
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

3. **Set Environment Variables**
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=document-embeddings
   ```

### 4. Heroku

**Prerequisites:**
- Heroku account
- Heroku CLI

**Steps:**

1. **Create Heroku App**
```bash
heroku create your-app-name
```

2. **Set Environment Variables**
```bash
heroku config:set OPENAI_API_KEY=your_openai_api_key
heroku config:set PINECONE_API_KEY=your_pinecone_api_key
heroku config:set PINECONE_ENVIRONMENT=your_pinecone_environment
heroku config:set PINECONE_INDEX_NAME=document-embeddings
```

3. **Deploy**
```bash
git push heroku main
```

**Configuration Files:**
- `Procfile` - Already configured
- `runtime.txt` - Python version
- `requirements.txt` - Dependencies

### 5. AWS Lambda (Serverless)

**Prerequisites:**
- AWS account
- Serverless Framework or AWS SAM

**Using Mangum for FastAPI on Lambda:**

1. **Install Mangum**
```bash
pip install mangum
```

2. **Create Lambda Handler**
```python
# lambda_handler.py
from mangum import Mangum
from main import app

handler = Mangum(app)
```

3. **Deploy with Serverless Framework**
```yaml
# serverless.yml
service: llm-query-retrieval

provider:
  name: aws
  runtime: python3.11
  environment:
    OPENAI_API_KEY: ${env:OPENAI_API_KEY}
    PINECONE_API_KEY: ${env:PINECONE_API_KEY}

functions:
  api:
    handler: lambda_handler.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
```

### 6. Google Cloud Run

**Prerequisites:**
- Google Cloud account
- gcloud CLI

**Steps:**

1. **Build Container**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/llm-query-system
```

2. **Deploy**
```bash
gcloud run deploy --image gcr.io/PROJECT_ID/llm-query-system \
  --platform managed \
  --set-env-vars OPENAI_API_KEY=your_key,PINECONE_API_KEY=your_key
```

### 7. DigitalOcean App Platform

**Prerequisites:**
- DigitalOcean account

**Steps:**

1. **Create App**
   - Go to DigitalOcean dashboard
   - Click "Create" → "Apps"
   - Connect your GitHub repository

2. **Configure App**
   ```
   Name: llm-query-retrieval-system
   Source: GitHub repository
   Branch: main
   Autodeploy: Yes
   ```

3. **Set Environment Variables**
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=document-embeddings
   ```

## 🔧 Post-Deployment Configuration

### 1. Verify Deployment

Test your deployed API:
```bash
curl -X GET "https://your-deployed-url.com/health" \
  -H "Authorization: Bearer YOUR_BEARER_TOKEN"
```

### 2. Test with Sample Data

```bash
curl -X POST "https://your-deployed-url.com/api/v1/hackrx/run" \
  -H "Authorization: Bearer YOUR_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

### 3. Monitor Performance

- Check response times (<30 seconds requirement)
- Monitor error rates
- Track API usage and costs

### 4. Set Up Custom Domain (Optional)

Most platforms support custom domains:
- Add CNAME record pointing to platform URL
- Configure SSL certificate
- Update API documentation

## 🚨 Troubleshooting

### Common Issues

1. **Cold Start Delays**
   - Use platform-specific warming strategies
   - Consider keeping one instance warm

2. **Memory Limits**
   - Optimize chunk sizes
   - Use streaming for large documents
   - Consider upgrading instance size

3. **Timeout Issues**
   - Increase platform timeout limits
   - Optimize query processing
   - Use async processing for large requests

4. **Environment Variables**
   - Verify all required variables are set
   - Check for typos in variable names
   - Ensure API keys are valid

### Platform-Specific Issues

**Vercel:**
- 10-second timeout limit for Hobby plan
- Consider Pro plan for longer timeouts

**Railway:**
- Check build logs for dependency issues
- Verify Procfile syntax

**Heroku:**
- Dyno sleeping on free tier
- Consider paid dynos for production

## 📊 Performance Optimization

1. **Caching**
   - Implement Redis for document caching
   - Use CDN for static assets

2. **Database Optimization**
   - Use connection pooling
   - Optimize Pinecone index settings

3. **Code Optimization**
   - Profile slow endpoints
   - Optimize embedding generation
   - Use batch processing where possible

## 🔒 Security Considerations

1. **API Keys**
   - Use environment variables
   - Rotate keys regularly
   - Monitor usage

2. **Rate Limiting**
   - Implement rate limiting
   - Use API gateways

3. **HTTPS**
   - Ensure HTTPS is enabled
   - Use proper SSL certificates

4. **Input Validation**
   - Validate all inputs
   - Sanitize file uploads
   - Check document sizes
