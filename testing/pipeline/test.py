import torch

from src.utils.save_load import load_model
from src.utils.config_funcs import load_yaml
from src.data.funcs import dataset_into_loader, train_test_split

def __test__(data, model_name):
    config_root = 'configs/dynamic/model_configs/hyperparametrs_search_result_config.yaml'
    params = load_yaml(config_root)
    _, val_data = train_test_split(data, resolution=params['resolution'], test_size=0.5)
    val_loader = dataset_into_loader(val_data, batch_size=params['batch_size'])
    test_loop(val_loader=val_loader, model_name=model_name, config_root=config_root)
    print(f'Тестирование окончено')

def test_loop(val_loader, model_name, config_root):
    device = 'cuda'
    model_params = load_yaml(config_root)
    model = load_model(params=model_params, model_name=model_name).to(device)
    
    y_preds = []
    y_trues = []
    test_acc = []

    model.eval()
    for X, y in val_loader:
        X = X.to(device)
        y = y.to(device)
        y_trues.extend(y.cpu().numpy())
        y_pred = model(X)
        y_pred = torch.argmax(y_pred, dim=1)
        y_preds.extend(y_pred.cpu().numpy())
        acc = sum(y == y_pred) / len(y)
        test_acc.append(acc.cpu().detach().numpy())

    

