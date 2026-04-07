# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.11-alpine

# ── Metadata ────────────────────────────────────────────────────────────────
LABEL maintainer="OpenEnv Submission"
LABEL description="AI Customer Support Resolution Environment"

# ── System deps ─────────────────────────────────────────────────────────────
RUN apk add --no-cache curl

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (cached layer) ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ───────────────────────────────────────────────────────
COPY . .

# ── Hugging Face Spaces runs as a non-root user ──────────────────────────────
RUN adduser -D -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ── Port ─────────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
