#!/usr/bin/env bash
# One entry point for every scenario in the README. Run `scripts/run_scenario.sh help` for usage.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"          # python interpreter (override e.g. PYTHON=python3.10)
GPUS="${GPUS:-}"                # number of GPUs for training (default: all visible)

usage() {
cat <<'USAGE'
Usage: scripts/run_scenario.sh <scenario> [options] [-- extra --set overrides, e.g. train.iterations=1000]

Training scenarios (config used):
  smoke         tiny model, 20 iterations           configs/examples/vsr_smoke_test.yaml
  standard      4x video SR, 256 px crops, T=5      configs/vsr_DiT.yaml
  small_gpu     128 px crops, T=3, 20M parameters   configs/examples/vsr_small_gpu.yaml
  single_frame  T=1 (image style training)          configs/examples/vsr_single_frame.yaml
  x2            2x upscaling                        configs/examples/vsr_x2.yaml
    --train DIR      training videos (frame folders / video files, searched recursively)   [required]
    --val_lq DIR     validation low-quality videos                                         [required]
    --val_gt DIR     validation ground truth (optional: omit to skip PSNR/LPIPS)
    --out DIR        output root (default: runs/<scenario>); results go to <out>/<timestamp>/
    --resume FILE    continue a run from ckpts/model_<iter>.pth (same scenario; still pass --train/--val_*)
    --init FILE      start a new run from these weights (model.ckpt_path)

Other commands:
  infer         restore videos           --input PATH --ckpt FILE [--scenario NAME|--config FILE] [--out DIR]
                                         [--scale 4] [--patch] (WeatherDiff-style patch aggregation)
  evaluate      PSNR/SSIM/TDE vs GT      --sr PATH --gt PATH [--metrics psnr,ssim,tde]
  demo-data     create a synthetic demo dataset in ./demo_data
  help          this text

Environment: PYTHON (interpreter), GPUS (training processes; >1 uses torchrun).
Examples:
  scripts/run_scenario.sh demo-data
  scripts/run_scenario.sh smoke --train demo_data/train --val_lq demo_data/val/lq --val_gt demo_data/val/gt
  scripts/run_scenario.sh infer --scenario smoke --ckpt runs/smoke/<ts>/ema_ckpts/ema_model_20.pth --input demo_data/val/lq --out results
  scripts/run_scenario.sh evaluate --sr results --gt demo_data/val/gt
USAGE
}

config_for() {
  case "$1" in
    smoke)        echo configs/examples/vsr_smoke_test.yaml ;;
    standard)     echo configs/vsr_DiT.yaml ;;
    small_gpu)    echo configs/examples/vsr_small_gpu.yaml ;;
    single_frame) echo configs/examples/vsr_single_frame.yaml ;;
    x2)           echo configs/examples/vsr_x2.yaml ;;
    *) echo "unknown scenario '$1' (smoke|standard|small_gpu|single_frame|x2)" >&2; exit 2 ;;
  esac
}

cmd="${1:-help}"; [ $# -gt 0 ] && shift || true

case "$cmd" in
  help|-h|--help) usage; exit 0 ;;

  demo-data) exec $PY scripts/make_demo_data.py --out demo_data "$@" ;;

  smoke|standard|small_gpu|single_frame|x2)
    cfg="$(config_for "$cmd")"; train=""; val_lq=""; val_gt=""; out="runs/$cmd"; resume=""; init=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --train) train="$2"; shift 2 ;;
        --val_lq) val_lq="$2"; shift 2 ;;
        --val_gt) val_gt="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --resume) resume="$2"; shift 2 ;;
        --init) init="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option $1 (see: scripts/run_scenario.sh help)" >&2; exit 2 ;;
      esac
    done
    sets=()
    [ -n "$train" ] || { echo "--train is required (also with --resume: the config only has a placeholder path)" >&2; exit 2; }
    sets+=("data.train.params.dir_paths=[$train]")
    [ -z "$val_lq" ] || sets+=("data.val.params.lq_path=$val_lq")
    if [ -n "$val_gt" ]; then sets+=("data.val.params.gt_path=$val_gt"); else sets+=("data.val.params.gt_path=null"); fi
    [ -z "$init" ] || sets+=("model.ckpt_path=$init")
    sets+=("$@")
    n="${GPUS:-$($PY -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"
    [ "$n" -ge 1 ] || n=1
    args=(main.py --cfg_path "$cfg" --save_dir "$out")
    [ -z "$resume" ] || args+=(--resume "$resume")
    if [ "$n" -gt 1 ]; then
      exec torchrun --standalone --nproc_per_node="$n" --nnodes=1 "${args[@]}" --set "${sets[@]}"
    else
      exec $PY "${args[@]}" --set "${sets[@]}"
    fi ;;

  infer)
    input=""; ckpt=""; cfg="configs/vsr_DiT.yaml"; out="results"; scale=4; patch=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --input) input="$2"; shift 2 ;;
        --ckpt) ckpt="$2"; shift 2 ;;
        --scenario) cfg="$(config_for "$2")"; shift 2 ;;
        --config) cfg="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --scale) scale="$2"; shift 2 ;;
        --patch) patch="--patch_restoration true"; shift ;;
        --) shift; break ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$input" ] && [ -n "$ckpt" ] || { echo "--input and --ckpt are required" >&2; exit 2; }
    chop=128; stride=96      # LQ pixels; 128 is a multiple of the LQ unit at 4x (64) and 2x (128)
    exec $PY inference_video.py -i "$input" -o "$out" --config_path "$cfg" --ckpt_path "$ckpt" \
        --scale "$scale" --chop_size "$chop" --chop_stride "$stride" $patch "$@" ;;

  evaluate)
    sr=""; gt=""; metrics="psnr,ssim,tde"
    while [ $# -gt 0 ]; do
      case "$1" in
        --sr) sr="$2"; shift 2 ;;
        --gt) gt="$2"; shift 2 ;;
        --metrics) metrics="$2"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
      esac
    done
    [ -n "$sr" ] && [ -n "$gt" ] || { echo "--sr and --gt are required" >&2; exit 2; }
    exec $PY evaluate_video.py -i "$sr" -r "$gt" --metrics "$metrics" ;;

  *) echo "unknown command '$cmd'" >&2; usage; exit 2 ;;
esac
