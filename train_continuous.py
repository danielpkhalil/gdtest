import os
import sys
import pprint
from pathlib import Path
import hydra
import torch
import wandb
import warnings
from omegaconf import OmegaConf
import random
import numpy as np

from models.trainer import get_trainer
from models.datasets import get_loaders
# Now we import ContinuousDiffusion instead of ByteNetDiffusion
from models.model.continuous_diffusion import GaussianDiffusion


def print_gpu_memory():
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # In MB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)   # In MB
        print(f"GPU Memory Allocated: {memory_allocated:.2f} MB")
        print(f"GPU Memory Reserved:  {memory_reserved:.2f} MB")

def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed
    torch.cuda.manual_seed_all(seed)  # All GPUs seed (if using multi-GPU)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="configs/training", config_name="continuous_test")
def main(config):
    Path(config.exp_name).mkdir(parents=True, exist_ok=True)

    root_dir = hydra.utils.get_original_cwd()
    os.chdir(root_dir)

    # Instantiate the continuous diffusion model from config
    model = hydra.utils.instantiate(config.model, _recursive_=False)

    # Set seed for reproducibility (optional)
    set_seed(config.train.seed)

    # If you have a checkpoint to load, you could do so here:
    # if config.ckpt_path is not None:
    #     state_dict = torch.load(config.ckpt_path)['state_dict']
    #     model.load_state_dict(state_dict)

    if config.train.ngpu > 0:
        print_gpu_memory()
        model.to(torch.device("cuda"))
        print_gpu_memory()

    train_loader, validation_loader = get_loaders(config)

    trainer = get_trainer(config)

    print("After getting training data:")
    print_gpu_memory()

    # Suppress annoying warnings if desired
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader,
        )

    # If desired, save the state dict for compatibility with EvoDiff or other tools:
    # save_path = os.path.join("checkpoints", config.exp_name)
    # best_checkpoint = os.path.join(save_path, "best_model.ckpt")
    # continuous_model = ContinuousDiffusion.load_from_checkpoint(best_checkpoint)
    # torch.save(continuous_model.network.state_dict(), os.path.join(save_path, "best_model_state_dict.pth"))


if __name__ == "__main__":
    main()
    sys.exit()
