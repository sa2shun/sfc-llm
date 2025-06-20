# Multi-stage build for optimization
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml* ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/csvs /app/cache /app/logs \
    && chmod 755 /app/models /app/csvs /app/cache /app/logs

# Create non-root user for security
RUN groupadd -r sfc-llm && useradd -r -g sfc-llm sfc-llm \
    && chown -R sfc-llm:sfc-llm /app
USER sfc-llm

# Set Python path and other environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default environment variables for production
ENV SFC_LLM_API_HOST=0.0.0.0
ENV SFC_LLM_API_PORT=9001
ENV SFC_LLM_EMBEDDING_DEVICE=cpu
ENV SFC_LLM_LOG_LEVEL=INFO
ENV SFC_LLM_API_REQUIRE_AUTH=true
ENV SFC_LLM_EMBEDDING_CACHE_SIZE=100
ENV SFC_LLM_SEARCH_CACHE_SIZE=100

# Expose port
EXPOSE 9001

# Health check with improved timing
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:9001/health || exit 1

# Default command with graceful shutdown
CMD ["python", "-m", "src.chat_server"]

# Development stage with additional tools
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black flake8 mypy

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/csvs /app/cache /app/logs

# Set environment for development
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SFC_LLM_LOG_LEVEL=DEBUG
ENV SFC_LLM_API_REQUIRE_AUTH=false

# Default development command
CMD ["python", "-m", "src.chat_server"]

# VLM-enabled stage
FROM base as vlm

# Install additional VLM dependencies
RUN pip install --no-cache-dir \
    torch torchvision \
    transformers[vision] \
    pillow \
    opencv-python-headless

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/csvs /app/cache /app/logs \
    && chmod 755 /app/models /app/csvs /app/cache /app/logs

# Create non-root user
RUN groupadd -r sfc-llm && useradd -r -g sfc-llm sfc-llm \
    && chown -R sfc-llm:sfc-llm /app
USER sfc-llm

# VLM-specific environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SFC_LLM_API_HOST=0.0.0.0
ENV SFC_LLM_API_PORT=9001
ENV SFC_LLM_VLM_MODEL=llava-hf/llava-v1.6-mistral-7b-hf

# Expose port
EXPOSE 9001

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:9001/health || exit 1

# VLM server command
CMD ["python", "-m", "src.vlm_chat_server"]