# Language Identification - Docker image
# Build after training: the `models/` directory must contain the artifacts.

FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project (includes trained models if present).
COPY . .

# Streamlit must see the app in the container working dir.
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Default command: start both the API and the Streamlit UI.
CMD ["bash", "start.sh"]
