"""Full-reference evaluation of restored videos (CPU friendly, no model downloads unless --metrics has lpips).

    python evaluate_video.py -i results/ -r gt_videos/ [--metrics psnr,ssim,tde] [--out_json metrics.json]

-i / -r: a video (file or frame folder) or a folder of videos. Restored and reference videos are paired by name
(`name.mp4` <-> `name.mp4`, `name/` <-> `name/`, and file/folder stems are compared, so `a.mp4` pairs with `a/`).

Metrics, per frame and averaged over the video (Y channel of BT.601 YCbCr, 8-bit, `--border` pixels cropped):
    psnr   peak signal-to-noise ratio
    ssim   structural similarity (skimage)
    lpips  learned perceptual distance (needs the lpips weights; run with a GPU/internet once)
    tde    temporal difference error, mean |(SR_{t+1} - SR_t) - (GT_{t+1} - GT_t)| in 8-bit levels: how well the
           frame-to-frame change matches the reference; flicker and temporal drift increase it (lower is better)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from datapipe.video_datasets import VideoReader, find_videos

METRICS = ('psnr', 'ssim', 'lpips', 'tde')


def to_y(frames):
    """list of h x w x 3 float RGB [0,1] -> T x h x w float Y in [0,255], quantised to 8 bit like saved results."""
    rgb = np.round(np.stack(frames, 0) * 255.0).clip(0, 255)
    return (65.481 * rgb[..., 0] + 128.553 * rgb[..., 1] + 24.966 * rgb[..., 2]) / 255.0 + 16.0


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def pair_videos(in_path, ref_path):
    sr = {(v.stem if v.is_file() else v.name): v for v in find_videos(in_path)}
    gt = {(v.stem if v.is_file() else v.name): v for v in find_videos(ref_path)}
    if len(sr) == 1 and len(gt) == 1:                     # single video each: pair regardless of names
        return [(next(iter(sr)), next(iter(sr.values())), next(iter(gt.values())))]
    missing = sorted(set(sr) - set(gt))
    if missing:
        print(f'warning: no reference for {missing}')
    return [(n, sr[n], gt[n]) for n in sorted(sr) if n in gt]


def evaluate_pair(sr_path, gt_path, metrics, border=0, max_frames=None, lpips_model=None):
    sr_reader, gt_reader = VideoReader(sr_path), VideoReader(gt_path)
    n = min(sr_reader.num_frames, gt_reader.num_frames, max_frames or 10 ** 9)
    if sr_reader.num_frames != gt_reader.num_frames:
        print(f'warning: {sr_path.name}: {sr_reader.num_frames} vs {gt_reader.num_frames} frames, using {n}')
    sr_frames, gt_frames = sr_reader.read(range(n)), gt_reader.read(range(n))
    h = min(sr_frames[0].shape[0], gt_frames[0].shape[0]); w = min(sr_frames[0].shape[1], gt_frames[0].shape[1])
    if sr_frames[0].shape[:2] != gt_frames[0].shape[:2]:
        print(f'warning: {sr_path.name}: size {sr_frames[0].shape[:2]} vs {gt_frames[0].shape[:2]}, cropping to {(h, w)}')
    sr_frames = [f[:h, :w] for f in sr_frames]; gt_frames = [f[:h, :w] for f in gt_frames]
    sr, gt = to_y(sr_frames), to_y(gt_frames)
    if border:
        sr, gt = sr[:, border:-border, border:-border], gt[:, border:-border, border:-border]

    out = {'frames': n}
    if 'psnr' in metrics:
        out['psnr'] = float(np.mean([psnr(a, b) for a, b in zip(sr, gt)]))
    if 'ssim' in metrics:
        from skimage.metrics import structural_similarity
        out['ssim'] = float(np.mean([structural_similarity(a, b, data_range=255.0) for a, b in zip(sr, gt)]))
    if 'tde' in metrics:
        out['tde'] = float(np.mean(np.abs(np.diff(sr, axis=0) - np.diff(gt, axis=0)))) if n > 1 else float('nan')
    if 'lpips' in metrics:
        import torch
        x = torch.from_numpy(np.stack(sr_frames, 0)).permute(0, 3, 1, 2) * 2 - 1
        y = torch.from_numpy(np.stack(gt_frames, 0)).permute(0, 3, 1, 2) * 2 - 1
        with torch.no_grad():
            out['lpips'] = float(lpips_model(x, y).mean())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_path', required=True, help='restored video(s)')
    parser.add_argument('-r', '--ref_path', required=True, help='reference (ground-truth) video(s)')
    parser.add_argument('--metrics', default='psnr,ssim,tde', help=f'comma separated subset of {METRICS}')
    parser.add_argument('--border', type=int, default=0, help='pixels cropped from every side before psnr/ssim/tde')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--out_json', default=None)
    args = parser.parse_args()
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    assert all(m in METRICS for m in metrics), f'unknown metric in {metrics}; choose from {METRICS}'

    lpips_model = None
    if 'lpips' in metrics:
        import lpips
        lpips_model = lpips.LPIPS(net='vgg').eval()

    pairs = pair_videos(args.in_path, args.ref_path)
    assert pairs, f'no matching restored/reference videos in {args.in_path} and {args.ref_path}'
    results = {}
    for name, sr_path, gt_path in pairs:
        results[name] = evaluate_pair(sr_path, gt_path, metrics, args.border, args.max_frames, lpips_model)
        print(f'{name}: ' + ', '.join(f'{k}={v:.4f}' if k != 'frames' else f'{k}={v}' for k, v in results[name].items()))
    mean = {m: float(np.nanmean([r[m] for r in results.values()])) for m in metrics}
    print('mean over %d video(s): ' % len(results) + ', '.join(f'{k}={v:.4f}' for k, v in mean.items()))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({'videos': results, 'mean': mean}, indent=2))


if __name__ == '__main__':
    main()
