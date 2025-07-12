# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create cache directories with proper permissions
RUN mkdir -p /app/cache/huggingface \
    && mkdir -p /app/cache/sentence_transformers \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app

# Set cache environment variables
ENV HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/cache/sentence_transformers

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies as root to avoid permission issues
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the default embedding model to avoid runtime permission issues
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || echo "Model download failed, will retry at runtime"

# Copy application code
COPY main.py .
COPY health_check.py .
COPY grounding.txt .

# Change ownership of all files to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose health check port (optional)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import requests; import os; requests.get(f'http://localhost:8080/health') if os.getenv('HEALTH_CHECK_ENABLED') == 'true' else exit(0)" || exit 1

# Run the application
CMD ["python", "main.py"] 