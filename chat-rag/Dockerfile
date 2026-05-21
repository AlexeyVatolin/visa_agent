FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends curl unzip make && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock Makefile ./

RUN uv sync --frozen --no-dev

RUN make chroma-db

COPY src/ src/
COPY data/ data/

ENV PYTHONPATH=/app/src

ARG APP_MODE=query_app

CMD ["sh", "-c", "uv run src/${APP_MODE}.py"]
