import os
import sys
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import urllib.request
import json

def get_available_llm(preferred_models=("qwen2.5:1.5b", "qwen2.5:3b", "qwen3:4b", "llama3.2:3b", "mistral:latest")):
    """Detect available Ollama models dynamically, falling back to an installed one."""
    try:
        req = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        data = json.loads(req.read().decode())
        installed = [m["name"].split(":")[0] for m in data.get("models", [])] + [m["name"] for m in data.get("models", [])]
        for pref in preferred_models:
            pref_base = pref.split(":")[0]
            if pref in installed or pref_base in installed:
                return pref
        # Return first non-embedding model
        for m in data.get("models", []):
            name = m["name"]
            if "embed" not in name:
                return name
    except Exception:
        pass
    return "gemma3:4b"

def load_document(file_path):
    """Load PDF or TXT document based on extension."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document file not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .pdf, .txt, .md")
    return loader.load()

def main():
    doc_path = "company_policy.pdf"
    if not os.path.exists(doc_path) and os.path.exists("company_policy.txt"):
        doc_path = "company_policy.txt"

    print(f"[*] Step 1: Loading document from '{doc_path}'...")
    documents = load_document(doc_path)
    print(f"    Loaded {len(documents)} page/document section(s).")

    print("[*] Step 2: Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"    Created {len(chunks)} text chunk(s).")

    print("[*] Step 3: Generating embeddings with nomic-embed-text...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    print("[*] Step 4: Storing vectors in FAISS local vector database...")
    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )
    print("    Vector store built successfully.")

    model_name = get_available_llm()
    print(f"[*] Step 5: Initializing ChatOllama with model '{model_name}'...")
    llm = ChatOllama(
        model=model_name,
        temperature=0
    )

    # Question to answer
    question = sys.argv[1] if len(sys.argv) > 1 else "Explain the authentication policy"
    print(f"\n[*] Question: {question}")

    print("[*] Step 6: Performing similarity search (k=3)...")
    docs = vector_db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""You are an enterprise document assistant.
Answer the user's question using ONLY the following context.
Be direct, professional, and factual.

Context:
{context}

Question:
{question}

If the answer is completely unavailable in the context, say that you cannot find it.""")

    chain = prompt | llm

    print("[*] Step 7: Generating answer with local LLM...\n")
    response = chain.invoke({
        "context": context,
        "question": question
    })

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response.content)

    print("\n" + "=" * 60)
    print("CITATIONS & SOURCES:")
    print("=" * 60)
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", doc_path)
        page = doc.metadata.get("page", 1)
        preview = doc.page_content.replace("\n", " ").strip()[:140]
        print(f"[{i}] File: {source} | Page: {page}")
        print(f"    Excerpt: \"{preview}...\"\n")

if __name__ == "__main__":
    main()
