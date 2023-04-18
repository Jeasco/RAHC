import sys
sys.path.append(".")

# also disable grad to save memory
import torch


import numpy as np
import yaml
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ


def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval(), sd



def vqgan_encoder(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  # print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  # xrec = model.decode(z)
  return z


if __name__ == '__main__':
    # "logs/vqgan_gumbel_f8/configs/model.yaml"
    config32x32 = load_config("logs/vqgan_gumbel_f8/configs/model.yaml", display=False)
    model32x32 = load_vqgan(config32x32, ckpt_path="logs/vqgan_gumbel_f8/checkpoints/last.ckpt", is_gumbel=True).to(
        DEVICE)
