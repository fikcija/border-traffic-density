# syntax=docker/dockerfile:1
FROM python:3.11-slim
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 180 --retries 10 "mlflow>=2.14,<3" && mkdir -p /mlflow/artifacts
EXPOSE 5000
