import argparse
from pathlib import Path
from omegaconf import OmegaConf

from sampler_video import VideoSampler
from utils.util_opts import str2bool


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str, required=True, help="Video file, frame folder, or folder of videos.")
    parser.add_argument("-o", "--out_path", type=str, required=True, help="Output folder.")
    parser.add_argument("--config_path", type=str, default="configs/vsr_DiT.yaml")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=None, help="Frames per window (default: training clip length).")
    parser.add_argument("--frame_overlap", type=int, default=2, help="Frames shared by consecutive windows.")
    parser.add_argument("--chop_size", type=int, default=128, help="Spatial tile size in LQ pixels (multiple of 64).")
    parser.add_argument("--chop_stride", type=int, default=96, help="Spatial tile stride in LQ pixels.")
    parser.add_argument("--patch_restoration", type=str2bool, const=True, default=None, nargs="?",
                        help="Override `patch_restoration.enabled` of the config (WeatherDiff-style patch aggregation).")
    parser.add_argument("--save_frames", type=str2bool, const=True, default=False, nargs="?", help="Also write PNG frames.")
    parser.add_argument("--fp32", type=str2bool, const=True, default=False, nargs="?", help="disable amp")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main():
    args = get_parser()
    configs = OmegaConf.load(args.config_path)
    configs.model.ckpt_path = str(args.ckpt_path)
    configs.diffusion.params.sf = args.scale
    sampler = VideoSampler(
            configs, sf=args.scale, use_amp=not args.fp32,
            chop_size=args.chop_size, chop_stride=args.chop_stride,
            num_frames=args.num_frames, frame_overlap=args.frame_overlap,
            padding_offset=max(configs.model.params.get('lq_size', 64), 64), seed=args.seed,
            patch_restoration=args.patch_restoration,
            )
    sampler.inference(args.in_path, args.out_path, save_frames=args.save_frames)


if __name__ == '__main__':
    main()
