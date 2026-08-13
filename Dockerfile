FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag/ ./rag/
COPY data/ ./data/

EXPOSE 8000
# Render/most PaaS inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn rag.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
