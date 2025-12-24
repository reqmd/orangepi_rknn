import numpy as np

from src.data.dataset import LabeledDataset
from .config_funcs import load_yaml, save_yaml
from src.models.SP import SP
from src.models.MLP32 import MLP32
from src.models.MLP16 import MLP16
from src.models.CNN import CNN
from src.models.MobileNet import MobileNet


def choise_model(data: LabeledDataset):
    load_root = 'configs/static/model_configs/st_mnd_config.yaml'
    save_root = 'configs/dynamic/model_configs/mnd_config.yaml'

    params = load_yaml(load_root)
    H, W = [], []
    for images, _ in data:
        H.append(images.size[0])
        W.append(images.size[1])

    size_0 = int(np.mean(H))
    size_1 = int(np.mean(W))
    
    H, W = size_0, size_1
    for idx, res in enumerate(params['size']):
        if H <= res and W <= res :
            end_resolutions = params['size'][idx:]
            end_models = params['model'][idx:]
            break

    if res <= H and res <= W:
        end_resolutions = [params['size'][1], params['size'][2], params['size'][3]]
        end_models = [params['model'][1], params['model'][2], params['model'][3]]
    print(f'Полученные разрешения и модели{end_resolutions}, {end_models}')
    save_yaml(save_root, params = {'models':end_models, 'resolutions':end_resolutions})

def match_case(params: dict):
    _model_params_ = params
    model_name = _model_params_['model']
    keys_to_remove = ['lr', 'batch_size', 'weight_decay', 'model', 'resolution']
    for key in keys_to_remove:
        del _model_params_[key]

    match model_name:
        case 'SP':
            model = SP(**_model_params_)
        case 'MLP16':
            model = MLP16(**_model_params_)
        case 'MLP32':
            model = MLP32(**_model_params_)
        case 'CNN':
            model = CNN(**_model_params_)
        case 'MobileNet':
            model = MobileNet(**_model_params_)
    return model

# def calculate_avg_size_per_class(dataset: Dataset, num_classes: int):
#     class_sizes = {i: [] for i in range(num_classes)}

#     for image, label in dataset:
#         h, w = image.shape[-2:]  
#         class_sizes[label].append((h, w))

#     avg_sizes = {}
#     max_avg_height = 0
#     max_avg_width = 0

#     def filter_outliers(data):
#             Q1 = np.percentile(data, 25)
#             Q3 = np.percentile(data, 75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR
#             return data[(data >= lower_bound) & (data <= upper_bound)]
    
#     for cls in class_sizes:
#         sizes = np.array(class_sizes[cls])
#         heights, widths = sizes[:, 0], sizes[:, 1]

#         filtered_heights = filter_outliers(heights)
#         filtered_widths = filter_outliers(widths)

#         avg_h = np.mean(filtered_heights)
#         avg_w = np.mean(filtered_widths)

#         avg_sizes[cls] = (avg_h, avg_w)

#         if avg_h > max_avg_height:
#             max_avg_height = avg_h
#         if avg_w > max_avg_width:
#             max_avg_width = avg_w

#     return avg_sizes, (max_avg_height, max_avg_width)
