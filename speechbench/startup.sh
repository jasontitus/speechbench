#!/bin/bash
# speechbench VM startup script.
#
# Runs as root on first boot of a Deep Learning VM (PyTorch). The DLVM image
# already has CUDA + a Python environment; we layer the ASR-specific stack
# on top, pull the source tarball + assignment from GCS, run the benchmark,
# upload logs, and shut the VM down.
#
# Required instance metadata:
#   bucket          gs://<bucket-name>      (no trailing slash)
#   run-id          <run id>
#   src-uri         gs://.../src/<run_id>.tar.gz
#   assignment-uri  gs://.../runs/<run_id>/assignments/<vm>.json
# Optional:
#   rerun           "true" to ignore existing results
#   dashscope-api-key  passed through to env

set -euo pipefail

LOG_FILE=/var/log/speechbench-startup.log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "▶ speechbench startup at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "▶ host=$(hostname)  user=$(whoami)"

md() {
  # Read instance metadata; absent keys return empty string silently.
  curl -fsS -o /dev/null -w "%{http_code}" -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
    > /tmp/_md_status 2>/dev/null || true
  if [ "$(cat /tmp/_md_status 2>/dev/null)" = "200" ]; then
    curl -fsS -H "Metadata-Flavor: Google" \
      "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null
  fi
}

BUCKET="$(md bucket)"
RUN_ID="$(md run-id)"
SRC_URI="$(md src-uri)"
ASSIGNMENT_URI="$(md assignment-uri)"
RERUN="$(md rerun)"
DASHSCOPE_KEY="$(md dashscope-api-key)"

if [ -z "$BUCKET" ] || [ -z "$RUN_ID" ] || [ -z "$SRC_URI" ] || [ -z "$ASSIGNMENT_URI" ]; then
  echo "✗ missing required metadata (bucket/run-id/src-uri/assignment-uri)"
  exit 2
fi

VM_NAME="$(hostname)"
LOG_REMOTE="${BUCKET}/runs/${RUN_ID}/logs/${VM_NAME}.startup.log"
RUNNER_LOG_REMOTE="${BUCKET}/runs/${RUN_ID}/logs/${VM_NAME}.runner.log"
FAILED_MARKER="${BUCKET}/runs/${RUN_ID}/logs/${VM_NAME}.failed"

# On any error, push the log + a .failed marker to GCS so the orchestrator
# can spot the failure on `speechbench status`.
on_error() {
  echo "✗ startup failed at line $1"
  gsutil -q cp "$LOG_FILE" "$LOG_REMOTE" || true
  echo "failed at line $1 host=$VM_NAME" | gsutil -q cp - "$FAILED_MARKER" || true
  shutdown -h +1 || true
}
trap 'on_error $LINENO' ERR

echo "▶ installing system audio libs"
# ffmpeg + libsndfile for audio decode. apt is fast on the DLVM image (already
# has its caches warmed). Belt-and-suspenders alongside the soundfile python
# package, in case any model wrapper falls back to ffmpeg-based decoding.
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg libsndfile1 >/dev/null 2>&1 || \
  echo "  ! apt install of ffmpeg/libsndfile failed (non-fatal)"

echo "▶ locating Python with PyTorch"
# The DLVM PyTorch image ships pytorch on the system python. Older variants
# put it under /opt/conda; newer ones (cu128/nvidia-570) ship it on the system
# python directly. Probe the common locations and pick the first that imports
# torch successfully.
PY=""
for cand in /opt/conda/bin/python3 /opt/conda/bin/python /opt/deeplearning/conda/bin/python3 /usr/bin/python3 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
      PY="$cand"
      echo "   using $cand ($($cand -c 'import torch; print(torch.__version__)'))"
      break
    fi
  fi
done
if [ -z "$PY" ]; then
  echo "   ! no python with torch found; falling back to /usr/bin/python3 and installing torch"
  PY=/usr/bin/python3
fi

echo "▶ downloading source tarball"
mkdir -p /opt/speechbench
cd /opt/speechbench
gsutil -q cp "$SRC_URI" /opt/speechbench/src.tar.gz
tar -xzf /opt/speechbench/src.tar.gz

echo "▶ installing requirements"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements-vm.txt

echo "▶ verifying CUDA"
"$PY" - <<'PYEOF'
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PYEOF

echo "▶ downloading assignment"
mkdir -p /opt/speechbench/state
gsutil -q cp "$ASSIGNMENT_URI" /opt/speechbench/state/assignment.json
echo "Assignment contents:"
cat /opt/speechbench/state/assignment.json

EXTRA_ARGS=""
if [ "${RERUN:-}" = "true" ]; then
  EXTRA_ARGS="--rerun"
fi

if [ -n "${DASHSCOPE_KEY:-}" ]; then
  export DASHSCOPE_API_KEY="$DASHSCOPE_KEY"
fi

echo "▶ running speechbench.runner"
set +e
PYTHONPATH=/opt/speechbench "$PY" -m speechbench.runner \
  --assignment-uri "$ASSIGNMENT_URI" \
  --bucket "$BUCKET" \
  --run-id "$RUN_ID" \
  --vm-name "$VM_NAME" \
  $EXTRA_ARGS 2>&1 | tee /var/log/speechbench-runner.log
RC=$?
set -e

echo "▶ runner exit code: $RC"
gsutil -q cp /var/log/speechbench-runner.log "$RUNNER_LOG_REMOTE" || true
gsutil -q cp "$LOG_FILE" "$LOG_REMOTE" || true

if [ $RC -ne 0 ]; then
  echo "runner exit=$RC host=$VM_NAME" | gsutil -q cp - "$FAILED_MARKER" || true
fi

echo "▶ self-deleting instance"
# Delete ourselves so the boot disk is freed (otherwise SSD quota fills up).
# `gcloud compute instances delete` requires the instance's own zone — read it
# from instance metadata.
ZONE=$(curl -fsS -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/zone" 2>/dev/null \
  | awk -F/ '{print $NF}')
PROJECT=$(curl -fsS -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/project/project-id" 2>/dev/null)
if [ -n "$ZONE" ] && [ -n "$PROJECT" ]; then
  echo "  zone=$ZONE project=$PROJECT name=$VM_NAME — deleting"
  gcloud compute instances delete "$VM_NAME" \
    --zone="$ZONE" --project="$PROJECT" --quiet --delete-disks=all || true
else
  echo "  could not read zone/project metadata; falling back to shutdown"
fi
# Belt-and-suspenders: shut down even if delete failed, so we stop billing.
shutdown -h +1
