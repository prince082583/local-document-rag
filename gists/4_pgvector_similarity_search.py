# Production Vector Retrieval with PostgreSQL pgvector and Cosine Similarity
import psycopg2
from pgvector.psycopg2 import register_vector
from langchain_ollama import OllamaEmbeddings

def search_enterprise_documents(query_text: str, top_k: int = 3, min_similarity: float = 0.5):
    """
    Computes query vector using nomic-embed-text and retrieves 
    top-k most semantically relevant chunks from PostgreSQL via HNSW index.
    """
    # 1. Generate query embedding locally
    embedder = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    query_vector = embedder.embed_query(query_text)

    # 2. Query PostgreSQL using pgvector cosine distance operator (<=>)
    # Cosine Similarity = 1 - (embedding <=> query_vector)
    conn = psycopg2.connect("postgresql://postgres:localdev@localhost:5432/localai")
    register_vector(conn)

    sql = """
        SELECT 
            id,
            content,
            metadata,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM documents
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (query_vector, query_vector, min_similarity, query_vector, top_k))
        records = cur.fetchall()

    conn.close()

    results = []
    for r in records:
        results.append({
            "id": r[0],
            "content": r[1],
            "metadata": r[2],
            "similarity": round(float(r[3]), 4)
        })

    return results
