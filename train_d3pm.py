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
from models.datasets import (
    get_loaders,
)
from models.model.d3pm_evodiff import ByteNetDiffusion

def print_gpu_memory(stage=""):
    """Prints current GPU memory usage with an optional stage description."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # In MB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)   # In MB
        print(f"[{stage}] GPU Memory Allocated: {memory_allocated:.2f} MB")
        print(f"[{stage}] GPU Memory Reserved:  {memory_reserved:.2f} MB")

def set_seed(seed):
    random.seed(seed)                  # Python random module
    np.random.seed(seed)               # NumPy random seed
    torch.manual_seed(seed)            # PyTorch CPU seed
    torch.cuda.manual_seed(seed)       # PyTorch GPU seed
    torch.cuda.manual_seed_all(seed)   # All GPUs seed (if using multi-GPU)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(config_path="configs/training", config_name="d3pm_test")
def main(config):
    # Create experiment directory
    Path(config.exp_name).mkdir(parents=True, exist_ok=True)

    # Move to root directory
    root_dir = hydra.utils.get_original_cwd()
    os.chdir(root_dir)

    print_gpu_memory("Before model instantiation")

    # Instantiate the model
    model = hydra.utils.instantiate(config.model, _recursive_=False)

    print_gpu_memory("After model instantiation")

    # Move model to GPU if required
    if config.train.ngpu > 0:
        print_gpu_memory("Before moving model to GPU")
        model.to(torch.device("cuda"))
        print_gpu_memory("After moving model to GPU")

    # Get data loaders
    train_loader, validation_loader = get_loaders(config)

    print_gpu_memory("After loading data")

    # Initialize trainer
    trainer = get_trainer(config)

    print_gpu_memory("After initializing trainer")

    # Suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings

        print_gpu_memory("Before training starts")

        # Start training
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader,
        )

        print_gpu_memory("After training ends")

    # Optional: Save model state
    # save_path = os.path.join("checkpoints", config.exp_name)
    # best_checkpoint = os.path.join(save_path, "best_model.ckpt")
    # lightning_model = ByteNetDiffusion.load_from_checkpoint(best_checkpoint)
    # torch.save(lightning_model.network.state_dict(), os.path.join(save_path, "best_model_state_dict.pth"))

if __name__ == "__main__":
    main()
    sys.exit()
