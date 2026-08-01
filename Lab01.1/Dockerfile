# =============================================================================
# Dockerfile for Movie Rating Prediction API
# DDM501 - Lab 1: First ML Product
# =============================================================================

# Base image
FROM python:3.10-slim

# Install system dependencies (curl for health check and build tools for Surprise)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Add health check (checking every 30s, timeout after 10s, start after 5s, retry 3 times)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Set the startup command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
