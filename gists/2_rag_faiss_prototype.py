# Minimal In-Memory RAG Prototype with FAISS and Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# 1. Load proprietary PDF
loader = PyPDFLoader("company_policy.pdf")
documents = loader.load()

# 2. Chunk document into semantic segments
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# 3. Embed & index locally in FAISS
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = FAISS.from_documents(chunks, embeddings)

# 4. Perform similarity search for user query
question = "Explain the password and authentication policy"
relevant_docs = vector_db.similarity_search(question, k=3)
context = "\n\n".join(doc.page_content for doc in relevant_docs)

# 5. Local LLM inference with grounded context
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
prompt = ChatPromptTemplate.from_template("""
You are an enterprise document assistant.
Answer the question using ONLY the provided context. If unsure, state that it is not documented.

Context:
{context}

Question:
{question}

Answer:
""")

chain = prompt | llm
response = chain.invoke({"context": context, "question": question})
print(response.content)
