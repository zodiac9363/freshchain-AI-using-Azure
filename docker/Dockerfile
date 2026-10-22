# FreshChain AI — Production Multi-Stage Dockerfile
# Stage 1: Dependency Builder
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Minimal Runtime Image
FROM python:3.12-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PORT=8000

WORKDIR /app

# Install runtime curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source code
COPY etl/ /app/etl/
COPY ml/ /app/ml/
COPY src/ /app/src/
COPY functions/ /app/functions/
COPY app/ /app/app/
COPY data/ /app/data/
COPY report/ /app/report/

# Non-root user for security compliance
RUN useradd -u 1001 -m freshchain && \
    chown -R freshchain:freshchain /app
USER freshchain

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command launches the FastAPI prediction engine
CMD ["uvicorn", "functions.local_api:app", "--host", "0.0.0.0", "--port", "8000"]
