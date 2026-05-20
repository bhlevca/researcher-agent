!#/bin/bash/bin 

cd /home/bogdan/ai/agent
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mrfakename/Z-Image-Turbo",
    resume_download=True,
    max_workers=4,   # stable; increase to 8 only if link is reliable
)
print("download complete")