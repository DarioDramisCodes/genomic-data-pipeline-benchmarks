# ==============================================================================
# BASE IMAGE: Use official Python slim image for a compact production footprint
# ==============================================================================
FROM python:3.12-slim

# Set system-level environment variables for Python and TensorFlow stability
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for compilation and numerical backend stability
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency manifest first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install the production-frozen python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the remaining project codebase and configuration assets
COPY . .

# Run the automated unit test suite to validate environment health before container launch
RUN python -m unittest discover -s tests

# Set the operational entrypoint to trigger the high-throughput processing pipeline
CMD ["python", "main.py"]
