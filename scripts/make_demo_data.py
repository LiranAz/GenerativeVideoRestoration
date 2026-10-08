"""Create a small synthetic video dataset so every scenario can be tried without downloading anything.

    python scripts/make_demo_data.py --out demo_data

Creates (frames are PNGs, one folder per video, plus one .mp4 to exercise video-file loading):
    demo_data/train/                 8 videos x 12 frames, 320x320 (HR)   -> use as training videos
    demo_data/val/gt/                2 videos x 6 frames, 256x256 (HR)    -> validation / evaluation ground truth
    demo_data/val/lq/                the same videos at 1/4 size (64x64)  -> 4x scenarios
    demo_data/val/lq_x2/             the same videos at 1/2 size (128x128)-> the 2x scenario

The content is moving smooth textures with a few shapes, i.e. it has temporal structure but is NOT realistic: use it
to check that the pipeline runs, not to judge restoration quality.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def make_video(rng, n_frames, size, pad=64):
    """Smooth random texture panned over time, with a drifting disc and colour change."""
    big = size + pad
    layers = []
    for scale, amp in ((8, 1.0), (16, 0.6), (32, 0.4), (64, 0.25)):
        small = rng.rand(big // scale + 2, big // scale + 2, 3).astype(np.float32)
        layers.append(amp * cv2.resize(small, (big, big), interpolation=cv2.INTER_CUBIC))
    tex = sum(layers)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    vx, vy = rng.uniform(-2.0, 2.0, size=2)
    cx, cy, r = rng.uniform(0.3, 0.7, size=2).tolist() + [rng.uniform(0.08, 0.15)]
    dx, dy = rng.uniform(-0.01, 0.01, size=2)
    tint = rng.uniform(0.8, 1.2, size=3)
    frames = []
    for t in range(n_frames):
        ox, oy = int(round(pad / 2 + vx * t)), int(round(pad / 2 + vy * t))
        frame = tex[oy:oy + size, ox:ox + size].copy() * (1.0 + 0.03 * t * (tint - 1.0))
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
        disc = ((xx - (cx + dx * t)) ** 2 + (yy - (cy + dy * t)) ** 2) < r ** 2
        frame[disc] = 1.0 - frame[disc]
        frames.append((frame.clip(0, 1) * 255).astype(np.uint8))
    return frames


def write_frames(frames, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(folder / f'{i:05d}.png'), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='demo_data')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    out, rng = Path(args.out), np.random.RandomState(args.seed)

    for i in range(8):
        frames = make_video(rng, 12, 320)
        if i < 2:   # two of the training videos are stored as .mp4 to exercise video-file loading
            out.joinpath('train').mkdir(parents=True, exist_ok=True)
            w = cv2.VideoWriter(str(out / 'train' / f'train_{i}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 12, (320, 320))
            for f in frames:
                w.write(f)
            w.release()
        else:
            write_frames(frames, out / 'train' / f'train_{i}')

    for i in range(2):
        gt = make_video(rng, 6, 256)
        write_frames(gt, out / 'val' / 'gt' / f'val_{i}')
        for name, size in (('lq', 64), ('lq_x2', 128)):
            lq = [cv2.resize(cv2.GaussianBlur(f, (0, 0), 0.6), (size, size), interpolation=cv2.INTER_AREA) for f in gt]
            lq = [np.clip(l.astype(np.float32) + rng.randn(*l.shape) * 2.0, 0, 255).astype(np.uint8) for l in lq]
            write_frames(lq, out / 'val' / name / f'val_{i}')
    print(f'demo data written to {out}/ (train, val/gt, val/lq, val/lq_x2)')


if __name__ == '__main__':
    main()
