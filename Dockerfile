FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if required for network/certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY static/ ./static/

EXPOSE 88

ENV PORT=88
ENV PGHOST=host.docker.internal
ENV PGPORT=5432
ENV PGUSER=postgres
ENV PGPASSWORD=localdev
ENV PGDATABASE=localai
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "88"]
