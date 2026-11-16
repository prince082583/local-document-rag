# Test Ollama Local Embeddings Dimension
# Model: nomic-embed-text (768 dimensions)
from langchain_ollama import OllamaEmbeddings

# Initialize local embedding model via Ollama
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Generate dense vector for query
sample_text = "Enterprise single sign-on authentication policy"
vector = embeddings.embed_query(sample_text)

print(f"Embedding successful! Dimensions: {len(vector)}")  # Output: 768
