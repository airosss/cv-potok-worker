"""Build a BS-RoFormer or Mel-Band RoFormer from its UVR / ZFTurbo yaml config and load the checkpoint (CPU).
Used by the worker and the build check. A config with num_bands is a Mel-Band RoFormer (e.g. the lead/backing karaoke model)."""
import torch
import yaml

from .bs_roformer import BSRoformer
from .mel_band_roformer import MelBandRoformer


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))


def build(ckpt, conf):
    cfg = yaml.load(open(conf), Loader=_Loader)
    m = (MelBandRoformer if "num_bands" in cfg["model"] else BSRoformer)(**cfg["model"])
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd, strict=True)
    return m.eval(), cfg
