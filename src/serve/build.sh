#!/usr/bin/env bash
# Assemble the build context from the repo, then build and run.
#
#   bash src/serve/build.sh          # build the image
#   bash src/serve/build.sh --run    # build, then run on :8000
set -euo pipefail
cd "$(dirname "$0")"

cp ../data/prepare.py ../data/roi.py .
cp ../../artifacts/model_int8.tflite .
cp ../../config/roi_regions.json .

docker build -t border-traffic:latest .
echo "image size: $(docker images border-traffic:latest --format '{{.Size}}')"

if [ "${1:-}" = "--run" ]; then
  docker run --rm -p 8000:8000 border-traffic:latest
fi
