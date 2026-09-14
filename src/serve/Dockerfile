# tflite-runtime only - no TensorFlow. ~200 MB against ~1 GB for a TF image.
# This is why the model was quantised: compression and deployment are one decision.
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preprocessing modules are copied verbatim from src/ rather than reimplemented.
# registry.py, auth.py and db.py are app.py's own imports - leaving any of them out
# builds an image that fails at startup on ModuleNotFoundError rather than at build.
COPY prepare.py roi.py app.py registry.py auth.py db.py ./
COPY model_int8.tflite roi_regions.json ./

ENV MODEL_PATH=model_int8.tflite \
    ROI_PATH=roi_regions.json \
    LABELS=no_traffic,light,high \
    INPUT_SIZE=224 \
    SERVE_DB_PATH=/app/serve-data/serve.db

# Accounts and the prediction log outlive the container only if this is mounted.
VOLUME /app/serve-data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
