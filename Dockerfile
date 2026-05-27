FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY webapp/requirements_web.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

RUN mkdir -p results webapp/results

ENV PYTHONPATH=/app
ENV CORPUS_PATH=/app/corpus
ENV RESULTS_DIR=/app/results

EXPOSE 7860

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "7860"]
