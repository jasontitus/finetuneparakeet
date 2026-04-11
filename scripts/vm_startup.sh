#!/bin/bash
# VM startup script for parakeet-tdt-0.6b-v3 Lithuanian fine-tuning.
#
# Runs as root on first boot of a Deep Learning VM (PyTorch). DLVM ships
# with CUDA + torch; we layer NeMo on top, pull the project tarball,
# build manifests, run baseline eval, train, run post eval, upload
# everything to GCS, and self-delete.
#
# Required instance metadata:
#   bucket          gs://<bucket>             (no trailing slash)
#   run-id          <run id>
#   src-uri         gs://.../source.tar.gz    (project tarball)
#   mode            smoke|full                (see below)
# Optional:
#   hf-token        HuggingFace token (for gated datasets — not needed
#                   for VP/FLEURS/shunyalabs as of this writing)
#
# Modes:
#   smoke:  CV25 LT only, ~2k-clip train cap, 1 epoch, 300 step max.
#           Fast sanity check that WER moves at all. Budget: ~$0.15-0.30.
#   full:   CV25 + VoxPopuli + FLEURS + shunyalabs, 5 epochs, no caps.
#           Budget: ~$3-5.
set -euo pipefail

LOG_FILE=/var/log/ft-startup.log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "▶ ft-parakeet startup at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "▶ host=$(hostname)"

md() {
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
MODE="$(md mode)"
HF_TOKEN="$(md hf-token)"

# GCS locations we'll use for resumable state. These hold the partial
# checkpoint tree that the background uploader keeps pushing and that
# a relaunched VM reads on startup to resume from where the last VM
# died (preemption, zone stockout, crash, etc.).
CKPT_PREFIX_GCS=""   # filled in below once PREFIX is known

if [ -z "$BUCKET" ] || [ -z "$RUN_ID" ] || [ -z "$SRC_URI" ] || [ -z "$MODE" ]; then
  echo "✗ missing required metadata (bucket/run-id/src-uri/mode)"
  exit 2
fi

VM_NAME="$(hostname)"
PREFIX="${BUCKET}/finetune/${RUN_ID}"
LOG_REMOTE="${PREFIX}/logs/${VM_NAME}.startup.log"
RUNNER_LOG_REMOTE="${PREFIX}/logs/${VM_NAME}.runner.log"
FAILED_MARKER="${PREFIX}/logs/${VM_NAME}.failed"
DONE_MARKER="${PREFIX}/logs/${VM_NAME}.done"
# Where incremental training state is mirrored. Re-running with the
# same RUN_ID will pull this back down at the start of the next VM's
# startup and resume training from the last checkpoint.
CKPT_PREFIX_GCS="${PREFIX}/state/checkpoints"

on_error() {
  echo "✗ startup failed at line $1"
  gsutil -q cp "$LOG_FILE" "$LOG_REMOTE" || true
  echo "failed at line $1 host=$VM_NAME mode=$MODE" | gsutil -q cp - "$FAILED_MARKER" || true
  # Don't self-delete on failure — leaves the disk around for 30 min debugging.
  shutdown -h +30 || true
}
trap 'on_error $LINENO' ERR

echo "▶ installing system audio libs (ffmpeg, libsndfile)"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg libsndfile1 >/dev/null 2>&1 \
  || echo "  ! apt install non-fatal failure"

echo "▶ locating python with torch"
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
if [ -z "$PY" ]; then PY=/usr/bin/python3; fi

echo "▶ fetching project tarball"
mkdir -p /opt/ftparakeet
cd /opt/ftparakeet
gsutil -q cp "$SRC_URI" /opt/ftparakeet/source.tar.gz
tar -xzf source.tar.gz
ls -la

# Check whether a previous VM for this RUN_ID already wrote checkpoints
# to GCS. If so, pull them down now so the training script can resume.
echo "▶ checking for resumable state at $CKPT_PREFIX_GCS"
mkdir -p /opt/ftparakeet/checkpoints/lt-ft
if gsutil -q ls "$CKPT_PREFIX_GCS/lt-ft/" >/dev/null 2>&1; then
  echo "  ✓ found prior state — restoring"
  gsutil -q -m cp -r "$CKPT_PREFIX_GCS/lt-ft/*" /opt/ftparakeet/checkpoints/lt-ft/ || \
    echo "  ! restore failed (non-fatal, will start from scratch)"
  echo "  local state after restore:"
  ls -la /opt/ftparakeet/checkpoints/lt-ft/ || true
  find /opt/ftparakeet/checkpoints/lt-ft/ -maxdepth 3 -name "*.ckpt" -printf "    %p  %s bytes\n" 2>/dev/null || true
else
  echo "  (no prior state — fresh run)"
fi

echo "▶ installing ML deps (nemo_toolkit[asr], jiwer, datasets, soundfile)"
"$PY" -m pip install --upgrade pip
# Install a recent NeMo + supporting libs. NeMo pulls in lightning, hydra,
# omegaconf, sentencepiece, etc. transitively. This step takes ~3-5 min.
"$PY" -m pip install --quiet \
  "nemo_toolkit[asr]>=2.0,<2.5" \
  "lightning>=2.2,<2.6" \
  "datasets>=2.18,<4" \
  "soundfile>=0.12" \
  "librosa>=0.10" \
  "jiwer>=3.0" \
  "huggingface_hub>=0.20" \
  "editdistance" \
  "transformers>=4.40,<4.60"

echo "▶ verifying CUDA + NeMo import"
"$PY" - <<'PYEOF'
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
import nemo
print(f"nemo={nemo.__version__}")
import nemo.collections.asr as nemo_asr
print("nemo.collections.asr OK")
PYEOF

# Reduce fragmentation-related OOMs. Recommended by PyTorch docs when
# a long training run hits OOM with a lot of "reserved but unallocated"
# memory — exactly what we saw in lt-full-20260409-063739.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "▶ PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

if [ -n "${HF_TOKEN:-}" ]; then
  mkdir -p /root/.cache/huggingface
  echo "$HF_TOKEN" > /root/.cache/huggingface/token
  export HF_TOKEN
  echo "▶ HF token installed"
fi

# ─── per-mode configuration ─────────────────────────────────────────────────

case "$MODE" in
  smoke)
    echo "▶ MODE=smoke (CV25 LT only, 1 full epoch, frozen encoder)"
    PREP_ARGS="--datasets cv25_lt"
    # With the encoder frozen, only ~60M params train — safe to run a
    # full CV25 epoch (~800 optimizer steps @ effective batch 16). Takes
    # ~6-10 min on L4 and actually gives us signal on whether the recipe
    # converges.
    TRAIN_ARGS="--epochs 1"
    TRAIN_MANIFEST="data/manifests/cv25_lt_train.json"
    VAL_MANIFEST="data/manifests/cv25_lt_dev.json"
    TEST_MANIFEST="data/manifests/cv25_lt_test.json"
    # Baseline + post eval on a larger sample so the reported numbers
    # are less noisy than the 300-clip first run.
    BASELINE_EVAL_CAP="--max-clips 500"
    POST_EVAL_CAP="--max-clips 500"
    PIPELINE_TIMEOUT=4500    # 75 minutes
    ;;
  full)
    echo "▶ MODE=full (CV25 + VoxPopuli + FLEURS + shunyalabs, 5 epochs)"
    PREP_ARGS="--datasets cv25_lt voxpopuli_lt fleurs_lt shunyalabs_lt"
    TRAIN_ARGS="--epochs 5"
    TRAIN_MANIFEST="data/manifests/ALL_train.json"
    VAL_MANIFEST="data/manifests/cv25_lt_dev.json"
    TEST_MANIFEST="data/manifests/cv25_lt_test.json"
    BASELINE_EVAL_CAP=""
    POST_EVAL_CAP=""
    # Full run: allow up to 16 hours before the script kills itself to
    # prevent runaway spend in case training stalls.
    PIPELINE_TIMEOUT=57600   # 16 hours
    ;;
  *)
    echo "✗ unknown mode: $MODE (expected smoke or full)"
    exit 3
    ;;
esac

# ─── pipeline ───────────────────────────────────────────────────────────────

RUN_LOG=/var/log/ft-runner.log

# Write the pipeline to a temp script. Easier to reason about than
# nested bash -c quoting, and lets us hard-timeout the whole thing.
PIPELINE_SH=/tmp/ft-pipeline.sh
cat > "$PIPELINE_SH" <<EOF
#!/bin/bash
set -e
cd /opt/ftparakeet

echo "▶ Step 1: build manifests"
"$PY" scripts/03_prepare_manifests.py $PREP_ARGS

echo "▶ Step 2: baseline eval (pretrained parakeet-tdt-0.6b-v3)"
"$PY" scripts/04_eval.py \\
  --model nvidia/parakeet-tdt-0.6b-v3 \\
  --manifest "$TEST_MANIFEST" \\
  --out results/baseline_cv25_lt_test \\
  $BASELINE_EVAL_CAP

echo "▶ Step 3: fine-tune"
"$PY" scripts/05_finetune.py \\
  --config configs/finetune_lt.yaml \\
  --model nvidia/parakeet-tdt-0.6b-v3 \\
  --out-dir checkpoints/lt-ft \\
  --train-manifest "$TRAIN_MANIFEST" \\
  --val-manifest "$VAL_MANIFEST" \\
  $TRAIN_ARGS

echo "▶ Step 4: post-training eval"
"$PY" scripts/04_eval.py \\
  --model checkpoints/lt-ft/finetuned.nemo \\
  --manifest "$TEST_MANIFEST" \\
  --out results/finetuned_cv25_lt_test \\
  $POST_EVAL_CAP
EOF
chmod +x "$PIPELINE_SH"

echo "▶ pipeline script contents:"
cat "$PIPELINE_SH"
echo ""

# Start a background uploader that mirrors the checkpoints/ directory
# to GCS every 3 minutes. This means an unexpected VM death loses at
# most a few minutes of training progress instead of the whole run.
# Also mirrors the runner log so we can see progress mid-run from the
# laptop.
UPLOADER_PID_FILE=/tmp/ft-uploader.pid
(
  while true; do
    sleep 180
    # Use rsync semantics so we only upload changed files.
    if [ -d /opt/ftparakeet/checkpoints/lt-ft ]; then
      gsutil -q -m rsync -r /opt/ftparakeet/checkpoints/lt-ft "$CKPT_PREFIX_GCS/lt-ft" 2>/dev/null || true
    fi
    if [ -f "$RUN_LOG" ]; then
      gsutil -q cp "$RUN_LOG" "$RUNNER_LOG_REMOTE" 2>/dev/null || true
    fi
  done
) &
UPLOADER_PID=$!
echo $UPLOADER_PID > "$UPLOADER_PID_FILE"
echo "▶ background uploader started (pid=$UPLOADER_PID, every 3min)"

set +e
timeout --signal=TERM --kill-after=60 "$PIPELINE_TIMEOUT" bash "$PIPELINE_SH" 2>&1 | tee "$RUN_LOG"
PIPELINE_RC=${PIPESTATUS[0]}
set -e
echo "▶ pipeline exit code: $PIPELINE_RC"
if [ "$PIPELINE_RC" -eq 124 ]; then
  echo "✗ pipeline timed out after ${PIPELINE_TIMEOUT}s"
fi

# Stop the uploader and do one final sync before the main upload block
# below. Belt-and-suspenders.
kill "$UPLOADER_PID" 2>/dev/null || true
if [ -d /opt/ftparakeet/checkpoints/lt-ft ]; then
  gsutil -q -m rsync -r /opt/ftparakeet/checkpoints/lt-ft "$CKPT_PREFIX_GCS/lt-ft" 2>/dev/null || true
fi

echo "▶ uploading results to $PREFIX"
gsutil -q -m cp -r results "$PREFIX/results" || true
gsutil -q -m cp -r checkpoints/lt-ft/finetuned.nemo "$PREFIX/checkpoints/finetuned.nemo" || true
gsutil -q cp "$RUN_LOG" "$RUNNER_LOG_REMOTE" || true
gsutil -q cp "$LOG_FILE" "$LOG_REMOTE" || true
echo "mode=$MODE host=$VM_NAME" | gsutil -q cp - "$DONE_MARKER" || true

echo "▶ self-deleting instance"
ZONE=$(curl -fsS -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/zone" 2>/dev/null \
  | awk -F/ '{print $NF}')
PROJECT=$(curl -fsS -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/project/project-id" 2>/dev/null)
if [ -n "$ZONE" ] && [ -n "$PROJECT" ]; then
  echo "  zone=$ZONE project=$PROJECT name=$VM_NAME — deleting"
  gcloud compute instances delete "$VM_NAME" \
    --zone="$ZONE" --project="$PROJECT" --quiet --delete-disks=all || true
fi
shutdown -h +1
