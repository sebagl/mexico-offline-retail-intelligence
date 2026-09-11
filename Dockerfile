# Mexico Offline Retail Intelligence - production image for Render's free plan.
# Independent demonstration analyzing open INEGI DENUE data; not endorsed by INEGI.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    PORT=8000

WORKDIR /app

# Dependencies first so application edits do not invalidate this layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model at build time so the first request never
# waits on a network fetch and the runtime filesystem stays read-only.
RUN python -c "from fastembed import TextEmbedding; import os; \
TextEmbedding(model_name=os.environ['EMBEDDING_MODEL'], cache_dir=os.environ['FASTEMBED_CACHE_PATH'])"

# Validated DENUE artifacts are baked in: the runtime never calls the DENUE
# API and needs no INEGI token.
COPY app ./app
COPY data/manifest.json data/denue_establishments.json data/denue_aggregates.json data/knowledge.json ./data/

RUN groupadd --system app && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=3)"

# Exactly one worker: the model, records and embedding matrix live in memory once.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
