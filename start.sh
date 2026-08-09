#!/usr/bin/env bash
# Prepend the venv's cuDNN libs so all sublibraries (libcudnn_cnn.so.9 etc.)
# are resolved from nvidia-cudnn-cu13==9.20.0 rather than the system 9.22.0,
# which would cause CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDNN_LIB="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="$CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$SCRIPT_DIR/.venv/bin/uvicorn" researcher.main:app \
    --host 0.0.0.0 --port 8000 --reload
