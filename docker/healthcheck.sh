#!/usr/bin/env bash
# Healthy means all THREE supervised processes are up, not just the API.
#
# This also closes the gap supervisord otherwise leaves: Docker supervises one
# container, so without probing each process a dead Prefect server or a crash-looping
# flow runner would still report the container healthy.
set -e

# 1. inference API
python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)"
# 2. Prefect API + UI
python -c "import urllib.request as u; u.urlopen('http://localhost:4200/api/health', timeout=3)"
# 3. the flow runner serves no port, so ask supervisord directly
supervisorctl -c /etc/supervisord.conf status flow-runner | grep -q RUNNING
