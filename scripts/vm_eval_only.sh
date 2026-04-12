#!/bin/bash
# VM startup script for eval-only run. No training.
# Clones the repo from GitHub, runs gcp_eval.sh, uploads results to GCS.
set -euo pipefail

LOG_FILE=/var/log/ft-eval.log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "▶ eval-only startup at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

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
VM_NAME="$(hostname)"
PREFIX="${BUCKET}/finetune/${RUN_ID}"

on_error() {
  echo "✗ failed at line $1"
  gsutil -q cp "$LOG_FILE" "${PREFIX}/logs/${VM_NAME}.eval.log" || true
  echo "failed at line $1" | gsutil -q cp - "${PREFIX}/logs/${VM_NAME}.failed" || true
  shutdown -h +5 || true
}
trap 'on_error $LINENO' ERR

DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg libsndfile1 git >/dev/null 2>&1 || true

echo "▶ cloning repo"
cd /opt
git clone --depth 1 https://github.com/jasontitus/finetuneparakeet.git
cd /opt/finetuneparakeet

# DLVM has python3 but no python symlink
ln -sf "$(which python3)" /usr/local/bin/python

# Remove numba-cuda from gcp_eval.sh — it's a WSL2-only workaround
# that breaks the DLVM's pre-installed numba (introduces a
# cuda.bindings dependency that doesn't exist on DLVM).
sed -i "s/'numba-cuda==0.15.1'//" /opt/finetuneparakeet/scripts/gcp_eval.sh || true

# Pull leaderboard eval script from GCS (not in the GitHub repo yet)
gsutil -q cp gs://safecare-maps-speechbench/corpora/scripts/12_eval_leaderboard.py \
  /opt/finetuneparakeet/scripts/12_eval_leaderboard.py || true

echo "▶ running gcp_eval.sh"
OUT_BASE="results/gcp_${RUN_ID}" bash scripts/gcp_eval.sh

echo "▶ running leaderboard-compatible eval (BasicTextNormalizer)"
pip install --quiet whisper-normalizer 2>/dev/null || true
LM_FILE="/opt/ftparakeet/data/lm/lt_token_4gram.arpa"
LB_OUT="results/gcp_${RUN_ID}"

# Fine-tuned model — greedy
echo "▶ leaderboard: FLEURS lt (ft, greedy)"
"$PY" scripts/12_eval_leaderboard.py \\
  --model sliderforthewin/parakeet-tdt-lt \\
  --dataset google/fleurs --config lt_lt --split test \\
  --text-field transcription \\
  --out "\$LB_OUT/lb_ft_fleurs_greedy"

echo "▶ leaderboard: VoxPopuli lt (ft, greedy)"
"$PY" scripts/12_eval_leaderboard.py \\
  --model sliderforthewin/parakeet-tdt-lt \\
  --dataset facebook/voxpopuli --config lt --split test \\
  --text-field normalized_text \\
  --out "\$LB_OUT/lb_ft_voxpopuli_greedy"

# Fine-tuned model — beam+LM (best config: beam=4, alpha=0.5)
if [ -f "\$LM_FILE" ]; then
  echo "▶ leaderboard: FLEURS lt (ft, beam+LM α=0.5)"
  "$PY" scripts/12_eval_leaderboard.py \\
    --model sliderforthewin/parakeet-tdt-lt \\
    --dataset google/fleurs --config lt_lt --split test \\
    --text-field transcription \\
    --lm "\$LM_FILE" --beam-size 4 --alpha 0.5 \\
    --out "\$LB_OUT/lb_ft_fleurs_beamlm"

  echo "▶ leaderboard: VoxPopuli lt (ft, beam+LM α=0.5)"
  "$PY" scripts/12_eval_leaderboard.py \\
    --model sliderforthewin/parakeet-tdt-lt \\
    --dataset facebook/voxpopuli --config lt --split test \\
    --text-field normalized_text \\
    --lm "\$LM_FILE" --beam-size 4 --alpha 0.5 \\
    --out "\$LB_OUT/lb_ft_voxpopuli_beamlm"
fi

# Baseline model — greedy (for comparison)
echo "▶ leaderboard: FLEURS lt (baseline, greedy)"
"$PY" scripts/12_eval_leaderboard.py \\
  --model nvidia/parakeet-tdt-0.6b-v3 \\
  --dataset google/fleurs --config lt_lt --split test \\
  --text-field transcription \\
  --out "\$LB_OUT/lb_baseline_fleurs_greedy"

echo "▶ leaderboard: VoxPopuli lt (baseline, greedy)"
"$PY" scripts/12_eval_leaderboard.py \\
  --model nvidia/parakeet-tdt-0.6b-v3 \\
  --dataset facebook/voxpopuli --config lt --split test \\
  --text-field normalized_text \\
  --out "\$LB_OUT/lb_baseline_voxpopuli_greedy"

echo "▶ uploading results"
gsutil -q -m cp -r "results/gcp_${RUN_ID}" "${PREFIX}/results/" || true
gsutil -q cp "$LOG_FILE" "${PREFIX}/logs/${VM_NAME}.eval.log" || true
echo "done" | gsutil -q cp - "${PREFIX}/logs/${VM_NAME}.done" || true

echo "▶ self-deleting"
ZONE=$(curl -fsS -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/zone" 2>/dev/null \
  | awk -F/ '{print $NF}')
PROJECT=$(curl -fsS -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/project/project-id" 2>/dev/null)
gcloud compute instances delete "$VM_NAME" \
  --zone="$ZONE" --project="$PROJECT" --quiet --delete-disks=all || true
shutdown -h +1
