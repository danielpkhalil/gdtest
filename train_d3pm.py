import os
import sys
import pprint
from pathlib import Path
import hydra
import torch
import wandb
import warnings
from omegaconf import OmegaConf
import sys
import random
import numpy as np

from models.trainer import get_trainer
from models.datasets import (
    get_loaders,
)
from models.model.d3pm_evodiff import ByteNetDiffusion

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
    Path(config.exp_name).mkdir(parents=True, exist_ok=True)
    
    root_dir = hydra.utils.get_original_cwd()
    # Change back to the root directory
    os.chdir(root_dir)

    model = hydra.utils.instantiate(config.model, _recursive_=False) #_recursive_=False

    #TODO: set a seed for reproducibility

    #TODO: could implement model loading from evodiff.pretrained.load_sequence_checkpoint
    #Probably not necessary if model training is fast

    # if config.ckpt_path is not None:
    #     state_dict = torch.load(config.ckpt_path)['state_dict']
    #     model.load_state_dict(state_dict)
    if config.train.ngpu > 0:
        model.to(torch.device("cuda"))

    train_loader, validation_loader = get_loaders(config)

    #TODO: decide if you should use deterministic behavior which slows down training
    #set_seed(config.train.seed)

    if True: #training diffusion with random timesteps
        #model.freeze_for_discriminative()
        trainer = get_trainer(config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # catch really annoying BioPython warnings
            
            trainer.fit(
                model=model,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader,
            )
        
            #save the lightning checkpoint as a state dict of the network that is compatible with EvoDiff - let's just do this during loading instead
            # save_path = os.path.join("checkpoints", config.exp_name)
            # best_checkpoint = os.path.join(save_path, "best_model.ckpt")
            # lightning_model = ByteNetDiffusion.load_from_checkpoint(best_checkpoint)
            # torch.save(lightning_model.network.state_dict(), os.path.join(save_path, "best_model_state_dict.pth"))

    # else:
    #     train_dl, valid_dl = [
    #         make_discriminative_loader(config, model, dl) for dl in [train_dl, valid_dl]
    #     ]

if __name__ == "__main__":    
    main()
    sys.exit()