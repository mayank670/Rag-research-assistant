# RAG Research Assistant

A production-quality AI Research Assistant using **Flask · Gemini · LangGraph · ChromaDB**.

Upload PDF documents and ask questions — the system retrieves the most relevant passages and generates answers with source citations.

## Architecture

```
User Question
      ↓
[LangGraph Node 1] Retrieve top-5 chunks from ChromaDB
      ↓
[LangGraph Node 2] Validate context (filter low-relevance chunks)
      ↓
[LangGraph Node 3] Generate answer with Gemini 1.5 Flash
      ↓
[LangGraph Node 4] Return answer + source citations
```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python · Flask |
| AI Model | Google Gemini 1.5 Flash |
| Embeddings | Gemini `embedding-001` |
| Vector Store | ChromaDB (persistent) |
| Workflow | LangGraph stateful pipeline |
| PDF Parsing | PyMuPDF (fitz) |

## Setup

### 1. Clone / unzip the project
```bash
cd rag-assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API key
```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key:
# GOOGLE_API_KEY=your_key_here
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 5. Run the app
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## Project Structure

```
rag-assistant/
├── app.py                  # Flask app + API routes
├── requirements.txt
├── .env.example
├── README.md
├── uploads/                # Temporary PDF storage
├── vectorstore/            # ChromaDB persistent store
├── templates/
│   └── index.html          # Full UI (split-panel layout)
├── static/
│   ├── style.css           # Design system
│   └── script.js           # Frontend logic
├── utils/
│   ├── pdf_loader.py       # PDF extraction + chunking
│   ├── embeddings.py       # Gemini embeddings + ChromaDB ops
│   ├── retriever.py        # Context formatting + prompt building
│   └── summarizer.py       # Document summarization
└── graph/
    └── workflow.py         # LangGraph RAG pipeline
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/api/upload` | Upload PDF files |
| GET | `/api/documents` | List uploaded documents |
| DELETE | `/api/documents/<name>` | Remove a document |
| POST | `/api/ask` | Ask a question (RAG pipeline) |
| POST | `/api/summarize` | Generate document summary |
| POST | `/api/chat/clear` | Clear chat session |
| GET | `/api/status` | Health check |

## Features

- ✅ Multi-PDF upload with drag & drop
- ✅ Semantic search via ChromaDB + Gemini embeddings
- ✅ LangGraph 4-node pipeline (retrieve → validate → generate → respond)
- ✅ Source citations with page numbers and relevance scores
- ✅ Document summary (executive summary, key topics, findings)
- ✅ Chat history with download
- ✅ Delete documents from vector store
- ✅ Real-time status indicator
