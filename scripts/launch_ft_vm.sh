#!/bin/bash
# Launch a GCP L4 spot VM to run Lithuanian fine-tuning end-to-end.
#
# Usage:
#   scripts/launch_ft_vm.sh <mode>
#     mode = smoke | full
#
# Env overrides (all optional):
#   PROJECT   gcp project               (default: safecare-maps)
#   BUCKET    gcs bucket (gs:// prefix) (default: gs://safecare-maps-speechbench)
#   ZONE      gcp zone                  (default: us-central1-a)
#   GPU       gcp accelerator type      (default: nvidia-l4)
#   MACHINE   gce machine type          (default: g2-standard-8)
#   RUN_ID    override run id           (default: derived from mode + timestamp)
#   HF_TOKEN  huggingface token         (default: empty)
#
# The script:
#   1. tars the project into /tmp/ftparakeet-src-<run>.tar.gz
#   2. uploads to ${BUCKET}/finetune/${RUN_ID}/source.tar.gz
#   3. creates an L4 spot VM with scripts/vm_startup.sh as its startup
#   4. prints the commands to watch logs + fetch results
#
# Watching logs (from your laptop):
#   gsutil cat gs://...finetune/<run>/logs/<vm>.startup.log
#   gsutil ls gs://...finetune/<run>/logs/              # pick the VM name
#
# Fetching results after the run:
#   gsutil -m cp -r gs://...finetune/<run>/results ./results/<run>
set -euo pipefail

MODE="${1:-}"
if [ -z "$MODE" ] || { [ "$MODE" != "smoke" ] && [ "$MODE" != "full" ]; }; then
  echo "usage: $0 smoke|full" >&2
  exit 2
fi

PROJECT="${PROJECT:-safecare-maps}"
BUCKET="${BUCKET:-gs://safecare-maps-speechbench}"
# Ordered list of zones to try until one has capacity. Cover every
# region where we have 1+ L4 spot quota (checked at 2026-04-09).
# Override with ZONES="zone1 zone2 ..." to force a specific order.
ZONES="${ZONES:-\
us-central1-a us-central1-b us-central1-c \
us-east1-c us-east1-d us-east1-b \
us-east4-c us-east4-a us-east4-b \
us-west1-a us-west1-b us-west1-c \
us-west4-a us-west4-b us-west4-c \
northamerica-northeast1-a northamerica-northeast1-b \
europe-west1-b europe-west1-c europe-west1-d \
europe-west2-a europe-west2-b \
europe-west3-a europe-west3-b \
europe-west4-a europe-west4-b europe-west4-c \
asia-east1-a asia-east1-b asia-east1-c \
asia-southeast1-a asia-southeast1-b asia-southeast1-c \
asia-northeast1-a asia-northeast1-b \
asia-northeast3-a asia-northeast3-b}"
GPU="${GPU:-nvidia-l4}"
MACHINE="${MACHINE:-g2-standard-8}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
HF_TOKEN="${HF_TOKEN:-}"

TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
RUN_ID="${RUN_ID:-lt-${MODE}-${TIMESTAMP}}"
VM_NAME="ftparakeet-${MODE}-${TIMESTAMP}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "▶ project: $PROJECT"
echo "▶ bucket:  $BUCKET"
echo "▶ zones:   $ZONES (will try in order until one has capacity)"
echo "▶ machine: $MACHINE + $GPU (spot)"
echo "▶ run id:  $RUN_ID"
echo "▶ vm:      $VM_NAME"
echo "▶ mode:    $MODE"
echo ""

# Build source tarball, excluding junk. Include configs, scripts, requirements,
# PLAN.md, pyproject.toml. Exclude data/, checkpoints/, .venv/, *.pyc, and the
# .git directory.
SRC_TAR="/tmp/ftparakeet-src-${RUN_ID}.tar.gz"
echo "▶ tarring project → $SRC_TAR"
tar -czf "$SRC_TAR" \
  --exclude='.venv' \
  --exclude='data' \
  --exclude='checkpoints' \
  --exclude='results' \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='.DS_Store' \
  PLAN.md README.md pyproject.toml requirements.txt configs scripts 2>/dev/null \
  || { echo "✗ tar failed"; exit 3; }
ls -lh "$SRC_TAR"

SRC_URI="$BUCKET/finetune/$RUN_ID/source.tar.gz"
echo "▶ uploading → $SRC_URI"
gsutil -q cp "$SRC_TAR" "$SRC_URI"

STARTUP_SCRIPT="$REPO_ROOT/scripts/vm_startup.sh"
if [ ! -f "$STARTUP_SCRIPT" ]; then
  echo "✗ startup script not found: $STARTUP_SCRIPT"; exit 4
fi

# Build the metadata string. Note metadata values can't contain commas;
# our values don't.
META="bucket=$BUCKET,run-id=$RUN_ID,src-uri=$SRC_URI,mode=$MODE"
if [ -n "$HF_TOKEN" ]; then
  META="$META,hf-token=$HF_TOKEN"
fi

# STANDARD = on-demand, SPOT = preemptible (~3x cheaper but stockout-prone).
TERMINATION_FLAG=""
if [ "$PROVISIONING_MODEL" = "SPOT" ]; then
  TERMINATION_FLAG="--instance-termination-action=DELETE"
fi

echo "▶ creating VM (provisioning: $PROVISIONING_MODEL, trying zones in order)"
CREATED_ZONE=""
SILENCED_ERRORS=""
for Z in $ZONES; do
  printf "  ▶ trying %s ... " "$Z"
  set +e
  gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT" \
    --zone="$Z" \
    --machine-type="$MACHINE" \
    --image-family="pytorch-2-7-cu128-ubuntu-2204-nvidia-570" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-size=150GB \
    --boot-disk-type=pd-balanced \
    --accelerator="type=$GPU,count=1" \
    --maintenance-policy=TERMINATE \
    --provisioning-model="$PROVISIONING_MODEL" \
    $TERMINATION_FLAG \
    --metadata-from-file="startup-script=$STARTUP_SCRIPT" \
    --metadata="$META" \
    --scopes=cloud-platform \
    --no-shielded-secure-boot >/tmp/gcloud-create.out 2>/tmp/gcloud-create.err
  RC=$?
  set -e
  if [ $RC -eq 0 ]; then
    CREATED_ZONE="$Z"
    echo "✓ created"
    break
  fi
  # Non-stockout errors (quota, permission, bad flags) stop immediately.
  if grep -qE "ZONE_RESOURCE_POOL_EXHAUSTED|stockout|does not have enough resources" /tmp/gcloud-create.err; then
    echo "stockout"
    continue
  fi
  echo "failed"
  cat /tmp/gcloud-create.err
  exit 5
done

if [ -z "$CREATED_ZONE" ]; then
  echo ""
  echo "✗ no zone had L4 $PROVISIONING_MODEL capacity."
  if [ "$PROVISIONING_MODEL" = "SPOT" ]; then
    echo "  Try again in 10-30 min (spot capacity fluctuates), or retry"
    echo "  with on-demand pricing:"
    echo "      PROVISIONING_MODEL=STANDARD RUN_ID=$RUN_ID bash scripts/launch_ft_vm.sh $MODE"
    echo "  On-demand L4 is ~\$0.85/hr vs ~\$0.28 spot — ~\$3-4 more for a 4h run."
  fi
  exit 6
fi
ZONE="$CREATED_ZONE"

cat <<EOF

▶ VM created: $VM_NAME in $ZONE

To follow the startup log (VM may take a minute before it's writing):
  gsutil cat "$BUCKET/finetune/$RUN_ID/logs/${VM_NAME}.startup.log" 2>/dev/null || echo "not yet"

The background uploader mirrors the runner log to GCS every 3 min, so
you can watch progress without SSHing:
  watch -n 30 'gsutil cat "$BUCKET/finetune/$RUN_ID/logs/${VM_NAME}.runner.log" 2>/dev/null | tail -40'

Or SSH-tail if the VM is still alive:
  gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" \\
    --command="sudo tail -f /var/log/ft-runner.log"

When it's done, two markers will appear:
  gsutil ls "$BUCKET/finetune/$RUN_ID/logs/"
  # ...done  or  ...failed

### If the spot VM gets preempted mid-training:

Just relaunch with the *same* RUN_ID — the new VM will pull the
incrementally-uploaded checkpoint from GCS and resume:

  RUN_ID=$RUN_ID bash scripts/launch_ft_vm.sh $MODE

Background uploader runs every 3 minutes, so at most ~3 min of
training progress is lost on preemption.

### Pull results locally:
  gsutil -m cp -r "$BUCKET/finetune/$RUN_ID/results" ./results/$RUN_ID
  gsutil cp "$BUCKET/finetune/$RUN_ID/checkpoints/finetuned.nemo" ./checkpoints/$RUN_ID.nemo

Run id: $RUN_ID
Bucket prefix: $BUCKET/finetune/$RUN_ID/
EOF
