# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.3 AS uv

FROM ubuntu:24.04

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/model-broker/.venv \
    PATH=/opt/model-broker/.venv/bin:$PATH \
    PYTHONPATH=/opt/model-broker/src

WORKDIR /opt/model-broker

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src

RUN groupadd --system model-broker \
    && useradd --system --gid model-broker --home-dir /nonexistent --shell /usr/sbin/nologin model-broker

USER model-broker
EXPOSE 8000

CMD ["uvicorn", "--factory", "model_broker.application:create_app", "--host", "0.0.0.0", "--port", "8000"]
