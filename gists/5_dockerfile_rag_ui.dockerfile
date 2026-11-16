# Containerized FastAPI & ChatGPT Web UI
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source & static assets
COPY server.py rag_postgres.py .
COPY static/ ./static/

# Expose internal application port
EXPOSE 88

# Connect to host Docker network for PostgreSQL and Ollama
ENV PGHOST=host.docker.internal
ENV PGPORT=5432
ENV PGDATABASE=localai
ENV PGUSER=postgres
ENV PGPASSWORD=localdev
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

# Launch FastAPI ASGI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "88"]
