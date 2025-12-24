import torch

from src.utils.config_funcs import save_yaml, load_yaml

def device_config(params_roots: list):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for root in params_roots:
        params = load_yaml(root)
        params['device'] = device
        save_yaml(root, params = params)