#!/bin/bash
# RAG Research Assistant — One-command setup script

set -e
echo ""
echo "🔬 RAG Research Assistant Setup"
echo "================================"

# Create virtualenv
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing dependencies..."

# Core packages
pip install -q flask werkzeug python-dotenv PyMuPDF PyPDF2

# Try google-generativeai (may fail on some platforms)
echo "Installing Google Gemini SDK..."
pip install -q "google-generativeai>=0.8.0" || {
    echo "  Trying alternate package name..."
    pip install -q google-genai
}

# ChromaDB
echo "Installing ChromaDB..."
pip install -q "chromadb>=0.5.0"

# LangGraph
echo "Installing LangGraph..."
pip install -q "langgraph>=0.2.0" "langchain-core>=0.2.0"

echo ""
echo "✅ All packages installed!"
echo ""

# .env check
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env — add your GOOGLE_API_KEY:"
    echo "   GOOGLE_API_KEY=your_key_here"
    echo ""
fi

echo "🚀 Run with:  python app.py"
echo "   Then open: http://localhost:5000"
echo ""
