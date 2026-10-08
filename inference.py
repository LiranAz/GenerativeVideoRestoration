import argparse
from pathlib import Path
from omegaconf import OmegaConf
from sampler import Sampler

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("-r", "--ref_path", type=str, default=None, help="GT path.")
    parser.add_argument("--mask_path", type=str, default="", help="Mask path for inpainting.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=1, help="Batch size.")
    parser.add_argument("--chop_bs", type=int, default=1, help="Chop batch size")
    parser.add_argument("--config_path", type=str, required=True, help="config path.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="checkpoint path.")
    parser.add_argument("--chop_size", type=int, default=512, choices=[512, 256, 64], help="Chopping forward.")
    parser.add_argument("--chop_stride", type=int, default=-1,help="Chopping stride.")
    parser.add_argument("--fp32", type=str2bool, const=True, default=False, nargs="?", help="disable amp")
    parser.add_argument("--task", type=str, default="realsr", choices=['realsr', 'faceir'])
    args = parser.parse_args()
    return args

def get_configs(args):
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()

    if args.task == 'realsr':
        if 'vqf8' in args.config_path:
            vqgan_path = 'weights/vq_f8.ckpt'
        else:
            vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
        configs = OmegaConf.load(args.config_path)
        ckpt_path = args.ckpt_path
    elif args.task == 'faceir':
        vqgan_path = ckpt_dir / f'ffhq512_vq_f8_dim8_face.pth'
        configs = OmegaConf.load(args.config_path)
        ckpt_path = args.ckpt_path
    else:
        raise TypeError(f"Unexpected task type: {args.task}!")

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.sf = args.scale
    if hasattr(configs, 'autoencoder'):
        configs.autoencoder.ckpt_path = str(vqgan_path)

    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_stride < 0:
        if args.chop_size == 512:
            chop_stride = (512 - 64) * (4 // args.scale)
        elif args.chop_size == 256:
            chop_stride = (256 - 32) * (4 // args.scale)
        elif args.chop_size == 64:
            chop_stride = (64 - 16) * (4 // args.scale)
        else:
            raise ValueError("Chop size must be in [512, 256]")
    else:
        chop_stride = args.chop_stride * (4 // args.scale)
    args.chop_size *= (4 // args.scale)
    print(f"Chopping size/stride: {args.chop_size}/{chop_stride}")
    return configs, chop_stride

def main():
    args = get_parser()
    configs, chop_stride = get_configs(args)

    sampler = Sampler(
                configs,
                sf=args.scale,
                chop_size=args.chop_size,
                chop_stride=chop_stride,
                chop_bs=args.chop_bs,
                use_amp=not args.fp32,
                seed=args.seed,
                padding_offset=max(configs.model.params.get('lq_size', 64), 64),
            )
    sampler.inference(args.in_path, args.out_path, bs=args.bs, noise_repeat=False)
    
    import evaluate
    evaluate.evaluate(args.out_path, args.ref_path, None)

if __name__ == '__main__':
    main()
