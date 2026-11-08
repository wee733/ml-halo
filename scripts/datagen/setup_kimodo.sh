#!/usr/bin/env bash
set -euo pipefail
download=0
case "${1:-}" in
    --download-models) download=1 ;;
    --help)
        echo "Usage: bash scripts/datagen/setup_kimodo.sh [--download-models]"
        echo "Builds pinned Kimodo into the project-local Python overlay using the existing workstation runtime."
        echo "--download-models checks access, then caches the four required Hugging Face repositories."
        exit 0
        ;;
    "") ;;
    *) echo "Unknown argument. Use --help." >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
    echo "Unknown arguments. Use --help." >&2
    exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
command -v cmake >/dev/null
command -v uv >/dev/null
mkdir -p "$HALO_LOCAL_ROOT/python"
uv pip install --python "$HALO_PYTHON" --target "$HALO_LOCAL_ROOT/python" --no-deps \
    -r "$HALO_ROOT/configs/datagen/kimodo_overlay.txt"
"$HALO_PYTHON" "$HALO_ROOT/scripts/datagen/check_kimodo.py"
if [[ $download == 1 ]]; then
    "$HALO_PYTHON" - <<'PY'
from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download

models = {
    "nvidia/Kimodo-G1-RP-v1": "config.yaml",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp": "config.json",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised": "adapter_config.json",
    "meta-llama/Meta-Llama-3-8B-Instruct": "config.json",
}
unavailable = []
for repo, filename in models.items():
    try:
        get_hf_file_metadata(hf_hub_url(repo, filename))
        print(f"Access available for {repo}", flush=True)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        unavailable.append(f"{repo} ({type(exc).__name__}, HTTP {status})")
if unavailable:
    raise SystemExit("Required model access is unavailable\n" + "\n".join(unavailable))
for repo in models:
    patterns = ["*.json", "*.safetensors", "tokenizer*"] if repo.startswith("meta-llama/") else None
    snapshot_download(repo, allow_patterns=patterns, max_workers=2)
    print(f"Cached {repo}", flush=True)
PY
fi
