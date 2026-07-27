# Aletheia demo dashboard — production image.
#
# Serves the public fleet-memory dashboard (the project plan §9.1). It runs the
# OFFLINE demo world standalone — no cloud credentials required to start — so the
# same image is deployable before CockroachDB Cloud + Bedrock are provisioned, and
# reports mode "offline-demo" honestly. The live-mode dependencies (psycopg,
# boto3) are installed too, so the identical image serves live fleet memory once a
# CockroachDB + Bedrock adapter is injected — no rebuild.
#
# Target: AWS App Runner (image-based) on port 8080; also runs on ECS Fargate or
# `docker run -p 8080:8080` unchanged. Build/push/deploy: infra/deploy_demoapp.sh.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEMOAPP_HOST=0.0.0.0 \
    DEMOAPP_PORT=8080

WORKDIR /app

# 1) Dependencies first, in their own layer, so a code-only change does not force a
#    full reinstall. This list MIRRORS pyproject.toml [project.dependencies] —
#    pyproject stays the single source of truth; keep the two in sync. httpx and
#    python-dotenv are pulled in transitively but pinned here for a reproducible layer.
RUN pip install \
    "psycopg[binary]>=3.2" \
    "boto3>=1.34" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "pydantic>=2.8" \
    "httpx>=0.27" \
    "python-dotenv>=1.0"

# 2) Source. .dockerignore keeps out tests, internal docs, .git, .venv, secrets.
COPY . .

# 3) Install the package itself (deps already present) so `import aletheia` metadata
#    exists; the app still runs from /app source, so scenarios/ and
#    docs/experiment_data/ resolve by relative path exactly as in local dev.
RUN pip install --no-deps .

# Least privilege: never run the public service as root (App Runner best practice).
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Container-level liveness (App Runner also health-checks /healthz independently).
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz').status==200 else 1)"

CMD ["python", "-m", "demoapp"]
