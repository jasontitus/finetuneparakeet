# finetuneparakeet

Fine-tune NVIDIA `parakeet-tdt-0.6b-v3` on Lithuanian ASR data.

See **[PLAN.md](PLAN.md)** for the full strategy, data sources, cost
estimates, and tiered execution plan.

## Quick tour

```
finetuneparakeet/
├── PLAN.md                     # design + data + cost doc
├── README.md                   # this file
├── requirements.txt            # (local dev only; VM installs its own stack)
├── configs/
│   └── finetune_lt.yaml        # hydra-style overrides for training
├── scripts/
│   ├── 02_extract_cv25.py      # local:   unpack CV25 LT tarball + split stats
│   ├── 03_prepare_manifests.py # vm+local: build NeMo JSON manifests
│   ├── 04_eval.py              # vm:      run a NeMo ASR model on a manifest
│   ├── 05_finetune.py          # vm:      run the fine-tune loop
│   ├── vm_startup.sh           # vm:      GCP DLVM bootstrap + pipeline driver
│   └── launch_ft_vm.sh         # local:   spin up an L4 spot VM, upload src, go
├── data/                       # local only (gitignored)
│   └── cv25_lt/                # extracted corpus + splits_summary.json
└── checkpoints/                # local only (gitignored)
```

## Running a job on GCP

We run everything on a 1×L4 spot VM in `safecare-maps / us-central1-a`.
The launch script tar-balls the project, uploads it to GCS, and creates
the VM. The VM boots, installs NeMo on top of the DLVM PyTorch image,
pulls the tarball, runs the pipeline end-to-end (manifest prep →
baseline eval → fine-tune → post eval), uploads results, and
self-deletes.

Two modes:

**Smoke test** — CV25 LT only, capped training, eval on 300 clips.
Budget ~$0.15-0.30. Goal: confirm the recipe actually moves WER
before spending real money.
```bash
bash scripts/launch_ft_vm.sh smoke
```

**Full run** — CV25 LT + VoxPopuli LT + FLEURS LT + shunyalabs LT,
5 epochs, full test eval. Budget ~$3-5. Kicks off after the smoke
test has validated the recipe.
```bash
bash scripts/launch_ft_vm.sh full
```

## Watching a run

The launch script prints the exact commands. Broadly:

```bash
# startup log (appears once the VM reaches its post-install phase):
gsutil cat gs://safecare-maps-speechbench/finetune/<run-id>/logs/<vm>.startup.log

# live runner log while training is going:
gcloud compute ssh <vm-name> --zone=us-central1-a \
    --command='sudo tail -f /var/log/ft-runner.log'

# a `.done` marker appears under logs/ when the whole thing finished:
gsutil ls gs://safecare-maps-speechbench/finetune/<run-id>/logs/
```

## Fetching results

```bash
gsutil -m cp -r \
    gs://safecare-maps-speechbench/finetune/<run-id>/results \
    ./results/<run-id>

gsutil cp \
    gs://safecare-maps-speechbench/finetune/<run-id>/checkpoints/finetuned.nemo \
    ./checkpoints/<run-id>.nemo
```

Each `results/<subset>/summary.json` has top-line WER/CER/RTFx for
that (model × manifest) combo. `results/<subset>/per_clip.jsonl` has
raw + normalized reference/hypothesis for every clip, so you can
inspect error types.

## Baseline numbers (speechbench, n=30 test clips)

| model                | WER  | CER   |
|----------------------|------|-------|
| parakeet-tdt-0.6b-v3 | 29.6%| 15.9% |
| whisper-large-v3     | 36.7%| 10.4% |
| fw-large-v3          | 39.2%| 11.4% |

parakeet-tdt-0.6b-v3 already beats every Whisper variant on LT — we're
aiming to pull its WER below ~20% with ~70h of LT training data.

## Local development

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install pandas soundfile tqdm jiwer google-cloud-storage pyyaml huggingface_hub pyarrow

# Extract CV25 LT tarball and print per-split stats:
python scripts/02_extract_cv25.py

# Build manifests from the local extract:
python scripts/03_prepare_manifests.py --datasets cv25_lt
```

The local venv deliberately does not install torch / NeMo — those only
live on the training VM.
