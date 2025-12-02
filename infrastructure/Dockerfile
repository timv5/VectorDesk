FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt is inside infrastructure/, but context is project root (..)
COPY infrastructure/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY llm-processor/qa.py ./qa.py
COPY embeddings-processor/index_docs.py ./index_docs.py
COPY embeddings-processor/data ./data

# start.sh is also inside infrastructure
COPY infrastructure/start.sh ./start.sh
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1

CMD ["./start.sh"]
