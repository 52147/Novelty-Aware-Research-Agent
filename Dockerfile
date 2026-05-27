FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies first (layer caching)
COPY requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt

# Pre-download the embedding model so first query is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy all project files
COPY . .

# Create results directory
RUN mkdir -p results webapp/results

# Set Python path so webapp can import from project root
ENV PYTHONPATH=/app
ENV CORPUS_PATH=/app/corpus
ENV RESULTS_DIR=/app/results

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "7860"]
