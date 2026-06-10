#!/usr/bin/env bash
set -euo pipefail

base_directory="${COMFYUI_BASE_DIRECTORY:-/data}"
listen="${COMFYUI_LISTEN:-0.0.0.0}"
port="${COMFYUI_PORT:-8188}"
database_url="${COMFYUI_DATABASE_URL:-sqlite:///$base_directory/user/comfyui.db}"

mkdir -p \
  "$base_directory/models/checkpoints" \
  "$base_directory/custom_nodes" \
  "$base_directory/input" \
  "$base_directory/output" \
  "$base_directory/temp" \
  "$base_directory/user" \
  "$base_directory/user/default"

cd /opt/ComfyUI

read -r -a extra_args <<< "${COMFYUI_ARGS:-}"
exec python main.py \
  --listen "$listen" \
  --port "$port" \
  --base-directory "$base_directory" \
  --database-url "$database_url" \
  --disable-auto-launch \
  "${extra_args[@]}"
