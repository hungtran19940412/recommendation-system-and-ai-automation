# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/metadata \
    /app/models/collaborative_filtering /app/models/content_based \
    /app/models/hybrid_model /app/models/metrics

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
