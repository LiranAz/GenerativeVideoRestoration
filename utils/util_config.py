"""Config loading with inheritance and command-line overrides.

A config may start with ``_base_: path/to/other.yaml`` (path relative to the repo root or to the config file);
it is merged on top of the base, so example configs only list what differs. Lists are replaced, not merged.
``--set key.sub=value`` style overrides (OmegaConf dotlist) are applied last.
"""
from pathlib import Path
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[1]


def load_config(path, overrides=None):
    path = Path(path)
    cfg = OmegaConf.load(path)
    base = cfg.pop('_base_', None)
    if base is not None:
        candidates = [Path(base), _ROOT / base, path.parent / base]
        base_path = next((c for c in candidates if c.is_file()), None)
        if base_path is None:
            raise FileNotFoundError(f'_base_ {base} of {path} not found (tried {[str(c) for c in candidates]})')
        cfg = OmegaConf.merge(load_config(base_path), cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
