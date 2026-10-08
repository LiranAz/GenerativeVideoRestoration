#!/usr/bin/env bash
# End-to-end demo in a few minutes: synthetic data -> smoke-test training (+validation) -> inference -> evaluation.
# Needs an NVIDIA GPU and the dependencies from requirements.txt. Results land in runs/quickstart/.
set -euo pipefail
cd "$(dirname "$0")/.."
R=scripts/run_scenario.sh

$R demo-data
$R smoke --train demo_data/train --val_lq demo_data/val/lq --val_gt demo_data/val/gt --out runs/quickstart

CKPT="$(ls -t runs/quickstart/*/ema_ckpts/*.pth | head -n 1)"
echo ">>> restoring demo_data/val/lq with $CKPT"
$R infer --scenario smoke --ckpt "$CKPT" --input demo_data/val/lq --out runs/quickstart/results
$R evaluate --sr runs/quickstart/results --gt demo_data/val/gt
echo ">>> done. Training logs/images: $(ls -dt runs/quickstart/20*/ | head -n 1)  restored frames: runs/quickstart/results"
