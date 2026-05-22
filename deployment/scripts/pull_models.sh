#!/usr/bin/env bash
# Pull PyTorch model checkpoints from the lab's Google Drive share into
# decider/model_weight/.
#
# Requires `gdown` (pip install gdown). Folder URL is shared internally —
# set GDRIVE_FOLDER_URL env var or edit below.

set -euo pipefail

HERE="$(cd "$(dirname "$0")"/.. && pwd)"
DEST="$HERE/src/core/decider/model_weight"
mkdir -p "$DEST"

URL="${GDRIVE_FOLDER_URL:-PASTE_LAB_SHARED_FOLDER_URL_HERE}"

if [ "$URL" = "PASTE_LAB_SHARED_FOLDER_URL_HERE" ]; then
  echo "Set GDRIVE_FOLDER_URL to the lab's shared model-weights folder, e.g.:"
  echo "  GDRIVE_FOLDER_URL=https://drive.google.com/drive/folders/XXXX bash $0"
  exit 1
fi

if ! command -v gdown >/dev/null; then
  echo "Installing gdown..."
  pip install --user gdown
fi

cd "$DEST"
gdown --folder "$URL" -O .

echo "Done. Checkpoints in: $DEST"
ls -lh "$DEST"
