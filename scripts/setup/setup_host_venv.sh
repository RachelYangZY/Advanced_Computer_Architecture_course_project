#!/usr/bin/env bash
set -eu

VENV="${HOME}/aca-venv"

if [ ! -x "${VENV}/bin/python" ]; then
  python3 -m virtualenv "${VENV}"
fi

"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install 'numpy==2.1.3' 'ai-edge-litert==2.2.0'
"${VENV}/bin/python" -c 'import numpy; from ai_edge_litert import interpreter; print("ENV_OK", numpy.__version__)'
