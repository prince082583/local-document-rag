import os
import sys
import shutil
import tempfile
import urllib.request
import json
from typing import List, Optional

import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="Enterprise RAG Assistant Chat UI")

# Environment configuration
PG_HOST = os.getenv("PGHOST", "host.docker.internal")
PG_PORT = os.getenv("PGPORT", "5432")
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASS = os.getenv("PGPASSWORD", "localdev")
PG_DB = os.getenv("PGDATABASE", "localai")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "qwen2.5:1.5b"

def get_db():
    """Connect to PostgreSQL with pgvector registered."""
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASS,
        dbname=PG_DB
    )
    try:
        register_vector(conn)
    except Exception:
        pass
    return conn

def init_db():
    """Ensure vector extension and table exist."""
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding VECTOR(768)
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_hnsw 
                ON documents USING hnsw (embedding vector_cosine_ops);
            """)
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] Warning during init_db: {e}")

# Call init_db on server startup
init_db()

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = DEFAULT_LLM_MODEL
    k: Optional[int] = 3

@app.get("/api/health")
def health_check():
    db_ok = False
    ollama_ok = False
    try:
        conn = get_db()
        conn.close()
        db_ok = True
    except Exception:
        pass

    try:
        req = urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        ollama_ok = (req.status == 200)
    except Exception:
        pass

    return {
        "status": "ok" if (db_ok and ollama_ok) else "degraded",
        "postgres": "connected" if db_ok else "unreachable",
        "ollama": "connected" if ollama_ok else "unreachable",
        "pghost": PG_HOST,
        "ollama_url": OLLAMA_BASE_URL
    }

@app.get("/api/models")
def list_models():
    try:
        req = urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        data = json.loads(req.read().decode())
        models = [m["name"] for m in data.get("models", []) if "embed" not in m["name"]]
        return {"models": models or [DEFAULT_LLM_MODEL]}
    except Exception:
        return {"models": [DEFAULT_LLM_MODEL]}

@app.get("/api/documents")
def get_documents():
    try:
        conn = get_db()
        docs = []
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COALESCE(metadata->>'source', 'Unknown') as src, COUNT(*) as chunks
                FROM documents
                GROUP BY metadata->>'source';
            """)
            rows = cur.fetchall()
            for r in rows:
                docs.append({
                    "name": os.path.basename(r[0]),
                    "full_path": r[0],
                    "chunks": r[1]
                })
        conn.close()
        return {"documents": docs}
    except Exception as e:
        return {"documents": [], "error": str(e)}

@app.delete("/api/documents")
def clear_documents():
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
            conn.commit()
        conn.close()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, and .md files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        texts = [chunk.page_content for chunk in chunks]
        vectors = embeddings.embed_documents(texts)

        conn = get_db()
        with conn.cursor() as cur:
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
                meta = {
                    "source": filename,
                    "page": chunk.metadata.get("page", 0),
                    "chunk_index": idx
                }
                cur.execute("""
                    INSERT INTO documents (content, metadata, embedding)
                    VALUES (%s, %s, %s);
                """, (chunk.page_content, Json(meta), vector))
            conn.commit()
        conn.close()
        return {"status": "success", "filename": filename, "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/chat")
async def chat(req: ChatRequest):
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty query.")

    try:
        # Step 1: Embed query
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        query_vector = embeddings.embed_query(question)

        # Step 2: pgvector similarity search
        conn = get_db()
        retrieved = []
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_vector, query_vector, req.k))
            rows = cur.fetchall()
            for r in rows:
                retrieved.append({
                    "id": r[0],
                    "content": r[1],
                    "metadata": r[2] or {},
                    "similarity": round(float(r[3]), 4)
                })
        conn.close()

        # Step 3: Format Context & Citations
        citations = []
        context_parts = []
        for i, r in enumerate(retrieved, 1):
            src = r["metadata"].get("source", "Unknown")
            page = r["metadata"].get("page", 0)
            score = r["similarity"]
            context_parts.append(f"[Source: {src} | Page {page}]:\n{r['content']}")
            citations.append({
                "index": i,
                "source": os.path.basename(src),
                "page": page,
                "similarity": score,
                "match_percent": int(max(0, min(100, score * 100))),
                "excerpt": r["content"].strip()
            })

        context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

        # Step 4: Generate Answer with Ollama
        llm = ChatOllama(
            model=req.model or DEFAULT_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0
        )

        prompt = ChatPromptTemplate.from_template("""You are an enterprise document assistant (like ChatGPT with full document awareness).
Answer the user's question clearly, professionally, and factually using ONLY the provided context.
Organize your response with clean markdown headings, bullet points, and bold terms where helpful.

Context:
{context}

Question:
{question}

If the answer is completely unavailable in the context, politely state that the information was not found in the ingested documents.""")

        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})

        return {
            "answer": response.content,
            "citations": citations,
            "model": req.model or DEFAULT_LLM_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>RAG Chat Assistant</h1><p>Static index.html not found.</p>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=88, reload=False)
