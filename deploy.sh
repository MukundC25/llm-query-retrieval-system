#!/bin/bash

# Deployment Script for LLM Query-Retrieval System
echo "🚀 Deploying LLM Query-Retrieval System..."

# Check if git is clean
if [[ -n $(git status --porcelain) ]]; then
    echo "📝 Committing changes..."
    git add .
    git commit -m "Deploy updates"
    git push origin main
fi

echo "✅ Code pushed to GitHub"
echo ""
echo "🌐 Next steps:"
echo "1. Go to: https://vercel.com/dashboard"
echo "2. Click 'Add New' → 'Project'"
echo "3. Import: MukundC25/llm-query-retrieval-system"
echo "4. Set environment variables:"
echo "   GEMINI_API_KEY=AIzaSyBT4LUdiN8qUOEl8xDhXAf8uIlWZTecwrk"
echo "   PINECONE_API_KEY=pcsk_6FvSNP_RFRePsH9Bg3K3CsU3Fn6BwxnAVEG9vA8oFYk2ceqiE3oVf9qvK99WsfnFR5RNpq"
echo "5. Deploy!"
echo ""
echo "🔗 Alternative platforms:"
echo "   Railway: https://railway.app"
echo "   Render: https://render.com"
echo "   Heroku: https://heroku.com"
