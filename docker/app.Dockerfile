# syntax=docker/dockerfile:1
# ONE image: training, orchestration and serving. supervisord runs the three
# processes. The Docker socket is not mounted anywhere - the flow imports the
# pipeline directly instead of starting sibling containers.
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/app-requirements.txt .
# TensorFlow is a ~280 MB wheel; the cache mount lets a dropped connection resume
# on retry instead of refetching, and its own layer keeps it out of later failures.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 180 --retries 10 "tensorflow>=2.16"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 180 --retries 10 -r app-requirements.txt

COPY src/ /app/src/
COPY config/roi_regions.json /app/config/roi_regions.json
COPY docker/supervisord.conf /etc/supervisord.conf
COPY docker/healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/healthcheck.sh

# Imports inside src/ are flat (`from prepare import ...`), so every package
# directory goes on the path. src/serve stays last: the API is started with
# --app-dir /app/src/serve and must resolve its own preprocessing copies first.
ENV PYTHONPATH=/app/src:/app/src/data:/app/src/training:/app/src/evaluation:/app/src/deployment:/app/src/serve

VOLUME ["/app/data", "/app/artifacts"]
EXPOSE 4200 8000
# Probes all three processes, so a dead one makes the CONTAINER unhealthy rather
# than hiding behind supervisord.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD bash /app/healthcheck.sh

CMD ["supervisord", "-c", "/etc/supervisord.conf"]
