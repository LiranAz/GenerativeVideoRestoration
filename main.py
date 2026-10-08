import argparse
from omegaconf import OmegaConf

from utils.util_common import get_obj_from_str
from utils.util_config import load_config
from utils.util_opts import str2bool

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./save_dir",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="resume from the save_dir or checkpoint",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/training/ffhq256_bicubic8.yaml",
            help="Configs of yaml file",
            )
    parser.add_argument(
            "--set",
            nargs="*",
            default=[],
            metavar="KEY=VALUE",
            help="Config overrides, e.g. --set train.iterations=100 'data.train.params.dir_paths=[/data/videos]'",
            )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()

    configs = load_config(args.cfg_path, args.set)    # supports `_base_:` inheritance and --set overrides

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume', ]:
            configs[key] = getattr(args, key)

    from utils.util_crop import resolve_crop
    resolve_crop(configs)    # crop-dependent values (model image_size, ...); no-op for configs without degradation.gt_size

    trainer = get_obj_from_str(configs.trainer.target)(configs)
    trainer.train()
