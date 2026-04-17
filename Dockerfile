# Diabetes Risk Prediction — Phase 2
# Multi-stage Docker build for Streamlit + FastAPI

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and artefacts
COPY app.py api.py ./
COPY data/ ./data/
COPY results/ ./results/
COPY explain/ ./explain/
COPY reports/ ./reports/

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Default: run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
