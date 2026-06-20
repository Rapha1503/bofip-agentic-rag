FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    BOFIP_AUTO_DOWNLOAD_ARTIFACTS=1 \
    BOFIP_PIPELINE_LOG=0 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && sed '/^torch==/d' requirements.txt > /tmp/requirements-no-torch.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-no-torch.txt

COPY app.py pyproject.toml ./
COPY src ./src
COPY docs/full_corpus_manifest.json ./docs/full_corpus_manifest.json

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
