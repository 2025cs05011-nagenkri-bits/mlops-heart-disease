# syntax=docker/dockerfile:1
#
# Heart Disease prediction service (Flask + scikit-learn pipeline).
# Designed to run on Cloud Run, Kubernetes, or any container runtime.

FROM python:3.9-slim

# Prevent Python from writing .pyc files and force stdout/stderr to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /app

# Install runtime dependencies first so this layer is cached when only code changes
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source package and the trained model artifact
COPY src/ ./src/
COPY models/ ./models/

# Cloud Run / generic runtime expect the service to listen on $PORT
EXPOSE 8080

# Production WSGI server (1 worker per pod so prometheus_client counters
# remain consistent within a process; horizontal scale comes from K8s replicas)
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 src.serving.app:app
