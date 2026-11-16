# How I Built a 100% Private, Enterprise-Grade Document RAG Assistant with PostgreSQL, pgvector, and Ollama

### *Zero cloud API bills. Zero data leakage. A complete architectural guide from an in-memory prototype to a containerized production system with a ChatGPT-style UI.*

**By Prince Jose** — *Senior Consultant @ Infosys | GenAI & Agentic AI Architect*  
*LinkedIn: [linkedin.com/in/prince-jose-bb7166104](https://www.linkedin.com/in/prince-jose-bb7166104) | GitHub: [github.com/prince082583](https://github.com/prince082583)*

---

### The End-to-End Pipeline

1. **Ingestion & Parsing:** Enterprise PDFs/TXTs ➔ `PyPDFLoader` text extraction.
2. **Chunking:** `RecursiveCharacterTextSplitter` (700-character segments, 100-character overlap).
3. **Embedding:** Local `nomic-embed-text` via Ollama generates dense 768-dimensional vectors.
4. **Indexing & Storage:** `PostgreSQL 18` + `pgvector` with an `HNSW` index for sub-millisecond approximate cosine similarity (`<=>`).
5. **Inference & Grounding:** Local `Qwen 2.5` model generates answers with verified page-level citations.
6. **Delivery:** ChatGPT dark-mode web application running inside Docker on port `8088:88`.

---

## The Enterprise Dilemma: Why Cloud LLMs Fail Privacy Audits

Almost every enterprise team wants to build internal AI assistants to query proprietary company policies, legal contracts, medical data, or codebase documentation.

However, the moment you propose sending confidential PDF documents to third-party cloud APIs (like OpenAI, Anthropic, or proprietary cloud endpoints), the conversation hits a brick wall with InfoSec, Legal, and Compliance teams:

1. **Data Sovereignty & Privacy:** Proprietary intellectual property leaves your network boundaries.
2. **Recurring Token Costs:** Monthly embedding and inference API bills scale unpredictably with document volume.
3. **Vendor Lock-in & Outage Risk:** If external APIs have latency spikes or downtime, your internal workflows halt.

**What if you could run the entire RAG pipeline 100% locally on your own workstation or private on-prem infrastructure?**

In this article, I will walk you step-by-step through how I built an enterprise-grade **Document RAG Assistant** that runs entirely offline using **PostgreSQL 18 + pgvector**, **Ollama (`nomic-embed-text` + `Qwen 2.5`)**, and packaged it inside Docker with a sleek **ChatGPT-style Web UI**.

---

## 🏗️ System Architecture & Tech Stack

Our local RAG system is engineered with 5 modular layers:

- **Ingestion & Parsing:** `PyPDFLoader` & `TextLoader` extract text, page numbers, and source metadata.
- **Chunking Engine:** `RecursiveCharacterTextSplitter` segments text into 700-character chunks with a 100-character sliding overlap.
- **Embedding Model:** `Ollama` (`nomic-embed-text`) generates dense 768-dimensional embeddings locally.
- **Vector Database:** `PostgreSQL 18` + `pgvector` with an `HNSW` index for sub-millisecond approximate nearest-neighbor search.
- **Inference Engine:** `Ollama` (`qwen2.5:1.5b` / `qwen2.5:3b`) running high-speed local inference on consumer hardware (NVIDIA RTX GPU) with zero cloud dependencies.
- **User Interface:** Modern ChatGPT dark-mode web application running inside Docker on port `8088:88`.

---

## Step 1: Setting Up Local Embeddings with Ollama

Traditional tutorials require installing heavy PyTorch packages (`sentence-transformers`) locally. While effective, a much cleaner approach is delegating both embeddings and chat inference to **Ollama**.

First, pull the specialized embedding model via your terminal:

`ollama pull nomic-embed-text`

This compact 274 MB model generates **768-dimensional** dense vectors. Next, pull our reasoning chat model:

`ollama pull qwen2.5:1.5b`

To verify that local embeddings are operating properly, we test vector generation with a quick script:

https://gist.github.com/prince082583/7812c1ec83f92dedc9beef7457d2dc64?file=1_ollama_embeddings_test.py

---

## Step 2: The Rapid In-Memory Prototype with FAISS

When building RAG systems, always validate document parsing, chunking parameters, and prompt templates with an in-memory vector store like **FAISS** before provisioning databases:

https://gist.github.com/prince082583/7812c1ec83f92dedc9beef7457d2dc64?file=2_rag_faiss_prototype.py

While FAISS is phenomenal for prototyping, in-memory indexes don't persist across application restarts, cannot scale horizontally across microservices, and lack relational filtering capabilities.

---

## Step 3: Production Upgrade to PostgreSQL 18 & pgvector

Why pay for niche standalone vector databases like Pinecone or Milvus when your trusted enterprise operational database—**PostgreSQL**—can natively handle vectors alongside metadata, relational tables, and ACID transactions?

### The PostgreSQL 18 Docker Gotcha

We spin up a dedicated PostgreSQL container with `pgvector` pre-installed:

`docker run -d --name local-postgres -e POSTGRES_PASSWORD=localdev -e POSTGRES_DB=localai -p 5432:5432 -v pgdata:/var/lib/postgresql pgvector/pgvector:pg18-trixie`

> ⚠️ **Real-World Engineering Gotcha:**  
> In PostgreSQL 18+ images, the official Docker distribution updated the internal storage layout to support `pg_ctlcluster`. If you mount volumes to `/var/lib/postgresql/data` (standard in PG 16/17), the container fails on startup. Always mount to `/var/lib/postgresql`.

### Designing the Schema & HNSW Index

In PostgreSQL, we enable the vector extension and create the `documents` table with an **HNSW (Hierarchical Navigable Small World)** index for approximate cosine similarity search:

https://gist.github.com/prince082583/7812c1ec83f92dedc9beef7457d2dc64?file=3_postgres_pgvector_schema.sql

### Performing Semantic Cosine Retrieval (`<=>`)

In `pgvector`, the `<=>` operator computes **Cosine Distance**:

$$\text{Cosine Similarity} = 1 - (\text{embedding} \Leftrightarrow \text{query\_vector})$$

Here is how our Python retrieval engine executes the vector query and returns similarity percentages:

https://gist.github.com/prince082583/7812c1ec83f92dedc9beef7457d2dc64?file=4_pgvector_similarity_search.py

---

## Step 4: Designing the ChatGPT-Style Web Interface

To make the assistant intuitive for enterprise users, we built a modern web UI mirroring ChatGPT's dark mode:

1. **Document Management Sidebar:** Displays all ingested documents with live chunk badges and a one-click deletion option.
2. **Drag & Drop Ingestion:** Users drop PDF, DOCX, or TXT files directly into the browser. The backend splits, embeds, and indexes them into PostgreSQL in seconds.
3. **Expandable Source Citations:** Underneath every answer, an accordion reveals the exact document name, page number, match percentage (e.g. `74% Match`), and the raw text excerpt.
4. **Markdown & Code Highlighting:** Formats responses with lists, tables, and highlighted syntax cleanly.

---

## Step 5: Containerizing with Docker on Port 8088:88

To ensure one-click deployment across internal developer machines and servers, we packaged the FastAPI backend and static UI assets into a Docker container:

https://gist.github.com/prince082583/7812c1ec83f92dedc9beef7457d2dc64?file=5_dockerfile_rag_ui.dockerfile

Run the container mapped to port `8088`:

`docker run -d --name rag-chat-ui -p 8088:88 -e PGHOST=host.docker.internal -e OLLAMA_BASE_URL=http://host.docker.internal:11434 rag-chat-ui:latest`

Open `http://localhost:8088` in your browser to start querying your enterprise documents!

---

## 📊 Performance Benchmarks (Running on RTX 4050 6GB)

- **Ingestion Speed:** 20-page technical PDF chunked and embedded in **~3.2 seconds**.
- **Vector Search Latency:** **~18ms** per query using PostgreSQL HNSW index.
- **Inference Time:** `Qwen 2.5 (1.5B)` generates complete grounded responses in **~1.4 seconds**.
- **Operational Cost:** **$0.00**. No API rate limits, no subscription tiers, and zero data leakage.

---

## 🎯 Key Takeaways for AI Engineers

1. **Ollama for Local Embeddings:** Decoupling embedding workloads (`nomic-embed-text`) from LLM inference keeps your stack lightweight without bloated local ML libraries.
2. **PostgreSQL is All You Need:** Before adding complex standalone vector stores to your infrastructure, leverage `pgvector`. It provides ACID compliance, rock-solid stability, and standard SQL tooling.
3. **Always Ground Answers with Citations:** In enterprise environments, trust is paramount. Always return document names, page numbers, and similarity metrics to eliminate hallucinations.

---

## 💻 Source Code & Full Repository

The entire open-source repository—including the FAISS prototype, PostgreSQL backend, FastAPI server, Docker configs, and Web UI—is available on GitHub:

👉 **[https://github.com/prince082583/local-document-rag](https://github.com/prince082583/local-document-rag)**

---

*If you enjoyed this architecture walkthrough, connect with me on [LinkedIn](https://www.linkedin.com/in/prince-jose-bb7166104) and star the project on [GitHub](https://github.com/prince082583) for more real-world Agentic AI and GenAI engineering guides!*
