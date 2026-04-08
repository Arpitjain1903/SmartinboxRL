# Use Python 3.11 slim image for performance and size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies (no heavy model pre-download — stays within 8 GB limit)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# ---------------------------------------------------------------------------
# Required environment variables (must be supplied at runtime via --env-file)
# ---------------------------------------------------------------------------
# API_BASE_URL  — OpenAI-compatible API endpoint
# MODEL_NAME    — LLM model identifier
# HF_TOKEN      — Hugging Face / API key

ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""
ENV OPENAI_API_KEY=""

# Runtime settings
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Healthcheck for Streamlit
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Default: run the Streamlit dashboard
# Override with: docker run ... python inference.py
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
