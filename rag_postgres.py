"""PostgreSQL + pgvector Document RAG Assistant
Enterprise Document Q&A pipeline with local embeddings and local LLMs via Ollama.
"""

import os
import sys
import json
import argparse
import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Database configuration defaults matching docker run command
PG_HOST = os.getenv("PGHOST", "localhost")
PG_PORT = os.getenv("PGPORT", "5432")
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASS = os.getenv("PGPASSWORD", "localdev")
PG_DB = os.getenv("PGDATABASE", "localai")

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

def get_available_llm(preferred_models=("qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "llama3.2:3b", "mistral:latest")):
    """Detect available Ollama models dynamically, falling back to an installed one."""
    import urllib.request
    try:
        req = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        data = json.loads(req.read().decode())
        installed = [m["name"].split(":")[0] for m in data.get("models", [])] + [m["name"] for m in data.get("models", [])]
        for pref in preferred_models:
            pref_base = pref.split(":")[0]
            if pref in installed or pref_base in installed:
                return pref
        for m in data.get("models", []):
            name = m["name"]
            if "embed" not in name:
                return name
    except Exception:
        pass
    return "qwen2.5:1.5b"

DEFAULT_LLM_MODEL = "qwen2.5:1.5b"

def get_db_connection():
    """Establish and return a connection to PostgreSQL with pgvector registered."""
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASS,
            dbname=PG_DB
        )
        try:
            register_vector(conn)
        except psycopg2.ProgrammingError:
            pass  # Extension not created yet; init_database will enable it
        return conn
    except psycopg2.OperationalError as e:
        print("\n" + "!" * 70)
        print("DATABASE CONNECTION ERROR:")
        print(f"Could not connect to PostgreSQL at {PG_HOST}:{PG_PORT}/{PG_DB}.")
        print("Please ensure Docker Desktop is running and start the container with:")
        print("""
docker run -d `
  --name local-postgres `
  -e POSTGRES_PASSWORD=localdev `
  -e POSTGRES_DB=localai `
  -p 5432:5432 `
  pgvector/pgvector:pg18-trixie
        """)
        print("!" * 70 + "\n")
        raise e

def init_database():
    """Create pgvector extension and documents table if not existing."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        print("[*] Enabling pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        print(f"[*] Creating documents table with VECTOR({EMBEDDING_DIM})...")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding VECTOR({EMBEDDING_DIM})
            );
        """)
        
        # Create HNSW index for ultra-fast approximate cosine similarity search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_hnsw 
            ON documents USING hnsw (embedding vector_cosine_ops);
        """)
        
        conn.commit()
    conn.close()
    print("[+] Database initialized successfully with pgvector and HNSW index.")

def load_and_split(file_path, chunk_size=700, chunk_overlap=100):
    """Load a document (PDF, TXT, MD) and split into chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format '{ext}'. Supported: .pdf, .txt, .md")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return chunks

def ingest_document(file_path):
    """Chunk, embed, and store document in PostgreSQL + pgvector."""
    init_database()
    print(f"[*] Loading and chunking document: {file_path}...")
    chunks = load_and_split(file_path)
    print(f"    Split into {len(chunks)} chunk(s).")

    print(f"[*] Generating embeddings using Ollama ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    conn = get_db_connection()
    with conn.cursor() as cur:
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            meta = {
                "source": chunk.metadata.get("source", file_path),
                "page": chunk.metadata.get("page", 1),
                "chunk_index": idx
            }
            cur.execute("""
                INSERT INTO documents (content, metadata, embedding)
                VALUES (%s, %s, %s);
            """, (chunk.page_content, Json(meta), vector))
        conn.commit()
    conn.close()
    print(f"[+] Successfully stored {len(chunks)} chunks in PostgreSQL + pgvector!")

def similarity_search(query_text, k=3):
    """Generate query embedding and perform cosine distance similarity search in pgvector."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    query_vector = embeddings.embed_query(query_text)

    conn = get_db_connection()
    results = []
    with conn.cursor() as cur:
        # <=> operator is cosine distance in pgvector: similarity = 1 - cosine_distance
        cur.execute(f"""
            SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_vector, query_vector, k))
        rows = cur.fetchall()
        for row in rows:
            results.append({
                "id": row[0],
                "content": row[1],
                "metadata": row[2],
                "similarity": float(row[3])
            })
    conn.close()
    return results

def query_rag(question, k=3, model_name=None):
    """Retrieve top-k contexts from PostgreSQL and answer with local Ollama LLM."""
    resolved_model = model_name or get_available_llm()
    print(f"[*] Querying pgvector for top-{k} relevant context chunks...")
    retrieved = similarity_search(question, k=k)

    if not retrieved:
        print("[!] No documents found in database. Ingest documents first using --ingest.")
        return

    context = "\n\n".join(
        f"[Chunk ID {r['id']} | Source: {r['metadata'].get('source')} (Page {r['metadata'].get('page')})]:\n{r['content']}"
        for r in retrieved
    )

    llm = ChatOllama(
        model=resolved_model,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""You are an enterprise document assistant.
Answer the user's question using ONLY the following context.
Be direct, professional, and factual.

Context:
{context}

Question:
{question}

If the answer is unavailable in the context, say that you cannot find it.""")

    chain = prompt | llm

    print(f"[*] Generating answer with local LLM ('{resolved_model}')...\n")
    response = chain.invoke({
        "context": context,
        "question": question
    })

    print("=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(response.content)

    print("\n" + "=" * 70)
    print("CITATIONS & SOURCES (from pgvector):")
    print("=" * 70)
    for i, r in enumerate(retrieved, 1):
        source = r["metadata"].get("source", "Unknown")
        page = r["metadata"].get("page", 1)
        score = r["similarity"]
        preview = r["content"].replace("\n", " ").strip()[:140]
        print(f"[{i}] {source} (Page {page}) - Cosine Similarity: {score:.4f}")
        print(f"    Excerpt: \"{preview}...\"\n")

def clear_documents():
    """Clear all documents from table."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
        conn.commit()
    conn.close()
    print("[+] All documents cleared from PostgreSQL.")

def main():
    parser = argparse.ArgumentParser(description="Document RAG Assistant with PostgreSQL + pgvector + Ollama")
    parser.add_argument("--init-db", action="store_true", help="Initialize pgvector extension and tables")
    parser.add_argument("--ingest", type=str, help="Ingest a document into PostgreSQL (PDF, TXT, MD)")
    parser.add_argument("--query", type=str, help="Ask a question against ingested documents")
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL, help="Ollama LLM model to use")
    parser.add_argument("--clear", action="store_true", help="Clear all stored documents")
    args = parser.parse_args()

    if args.init_db:
        init_database()
    elif args.clear:
        clear_documents()
    elif args.ingest:
        ingest_document(args.ingest)
    elif args.query:
        query_rag(args.query, model_name=args.model)
    else:
        # Default run demo: ingest company_policy.pdf if exists and query
        target_doc = "company_policy.pdf" if os.path.exists("company_policy.pdf") else "company_policy.txt"
        print(f"[*] Running default workflow with '{target_doc}'...")
        ingest_document(target_doc)
        print("\n" + "-" * 70)
        query_rag("Explain the authentication policy", model_name=args.model)

if __name__ == "__main__":
    main()
