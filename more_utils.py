import numpy as np 
import torch
import random
import os

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def save_model(model, optimizers, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optims': [optimizer.state_dict() for optimizer in optimizers],  #sst, para, sts optimizers as list
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"Saved the model to {filepath}!")