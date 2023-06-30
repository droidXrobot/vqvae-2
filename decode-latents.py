import torch
import numpy as np
from pathlib import Path
from vqvae2 import VQVAE, VQVAE2
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__file__)
set_seed(0xAAAA)

@torch.no_grad()
@torch.inference_mode()
def main(cfg: DictConfig, args):
    logger.info("Loaded Hydra config:")
    logger.info(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Initialising VQVAE model.")
    net = VQVAE2.build_from_config(
        cfg.vqvae.model, codebook_gumbel_temperature=0.0, codebook_init_type="kaiming_uniform", codebook_cosine=False
    )
    net.load_state_dict(torch.load(args.model_path))
    net.eval()
    net = net.to(device)

    in_dir = args.in_dir
    if isinstance(in_dir, str):
        in_dir = Path(in_dir)
    out_dir = args.out_dir
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    out_dir.mkdir()

    logging.info("Decoding latents.")
    for latent_file in in_dir.glob("*.npy"):
        latent = np.load(latent_file)
        latent = torch.tensor(latent, dtype=torch.long).to(device)

        decoded_data = net.decode(latent)

        decoded_data = decoded_data.cpu().numpy()

        np.save(out_dir / f"{latent_file.stem}_decoded.npy", decoded_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("in_dir", type=str, help="Directory containing the latent numpy files.")
    parser.add_argument("out_dir", type=str, help="Directory where the decoded numpy files will be saved.")
    args = parser.parse_args()

    hydra.initialize(version_base=None, config_path="config")
    cfg = hydra.compose(config_name="config", overrides=[f"data={args.config_name}", f"vqvae={args.config_name}"])

    main(cfg, args)
