# Azure AI Services & OpenAI Q&A (Streamlit)

Streamlit RAG app that answers questions about the `azure-ai-services-openai_may2024.pdf` document. It loads a prebuilt Chroma vector store, embeds with `thenlper/gte-large` (1024-dim), retrieves relevant chunks, and generates answers via a Groq LLM (`openai/gpt-oss-20b`).

## Requirements
- Python 3.9+.
- Local assets (must exist alongside the app):
  - `azure-ai-services-openai_may2024.pdf`
  - `azure-ai-services-openai_may2024_db/` (prebuilt Chroma DB; no auto-rebuild in the app)
- See `requirements.txt` (CPU torch and related deps):
  - `--extra-index-url https://download.pytorch.org/whl/cpu`
  - `streamlit`, `groq`, `langchain`, `langchain-community`, `langchain-text-splitters`, `chromadb`, `sentence-transformers`, `pypdf`, `tqdm`, `tiktoken`, `torch==2.9.1+cpu`

## Setup (Windows PowerShell)
```powershell
cd "c:\Users\wilke\Coding Projects\Azure Solution Architect"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run locally
```powershell
cd "c:\Users\wilke\Coding Projects\Azure Solution Architect"
.\.venv\Scripts\python.exe -m streamlit run AzureSolArcapp.py
```

## Usage
1) Ensure the PDF and `azure-ai-services-openai_may2024_db` folder are present in the app directory (required; the app does not rebuild the DB).  
2) Launch the app and enter your Groq API key in the sidebar.  
3) Ask a question in the text area, then click **Send** (or use Ctrl+Enter).  
4) The app retrieves context from the vector store and returns a grounded answer.

## Deployment notes
- Deploy the Chroma DB folder with the app (or provide a remote/persistent store). If the DB is missing, the app will stop with an error instead of rebuilding.  
- The embedding model is fixed to `thenlper/gte-large` to match the shipped DB dimensions. If you rebuild with a different model, rebuild and redeploy the DB accordingly.  
- If running outside Streamlit Cloud, bind to the provided port/host when required (e.g., `streamlit run ... --server.port $PORT --server.address 0.0.0.0`).***
