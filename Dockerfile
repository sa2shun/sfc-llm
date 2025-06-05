FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/csvs

# Set Python path
ENV PYTHONPATH=/app

# Default environment variables
ENV SFC_LLM_API_HOST=0.0.0.0
ENV SFC_LLM_API_PORT=9001
ENV SFC_LLM_EMBEDDING_DEVICE=cpu

# Expose port
EXPOSE 9001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9001/health || exit 1

# Default command
CMD ["python", "-m", "src.chat_server"]