# Multi-stage build for optimized Docker image
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY src/ ./src/
COPY requirements.txt ./
COPY README.md ./

# Create necessary directories
RUN mkdir -p results data/sample && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set Python path to include user packages
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import tensorflow as tf; print('Health check passed')" || exit 1

# Default command
CMD ["python", "src/quick_demo.py"]

# Metadata
LABEL maintainer="saazzam@ttu.edu"
LABEL version="1.0"
LABEL description="Card Classification CNN with Transfer Learning" 