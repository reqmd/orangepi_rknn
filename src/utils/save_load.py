import torch

from src.utils.funcs import match_case

def load_model(params: dict, model_name = None):
    model = match_case(params=params)
    if model_name != None:
        print('Веса модели загружены')
        model.load_state_dict(torch.load(model_name, weights_only=True))
    return model

def save_model(model_state_dict, model_name):
    torch.save(model_state_dict, model_name) 