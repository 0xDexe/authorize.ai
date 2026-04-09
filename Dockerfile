FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# If a pre-trained model is bundled in the repo, it will already be at data/models/.
# Alternatively mount it via docker-compose volumes (see docker-compose.yml).

EXPOSE $PORT

HEALTHCHECK CMD curl --fail http://localhost:$PORT/_stcore/health || exit 1

CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true
