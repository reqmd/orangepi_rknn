import numpy as np
from torch.utils.data import Dataset
import yaml

from models import MLP, MobileNet, SP, CNN
def calculate_avg_size_per_class(dataset: Dataset, num_classes: int):
    class_sizes = {i: [] for i in range(num_classes)}

    for image, label in dataset:
        h, w = image.shape[-2:]  
        class_sizes[label].append((h, w))

    avg_sizes = {}
    max_avg_height = 0
    max_avg_width = 0

    def filter_outliers(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data >= lower_bound) & (data <= upper_bound)]
    
    for cls in class_sizes:
        sizes = np.array(class_sizes[cls])
        heights, widths = sizes[:, 0], sizes[:, 1]

        filtered_heights = filter_outliers(heights)
        filtered_widths = filter_outliers(widths)

        avg_h = np.mean(filtered_heights)
        avg_w = np.mean(filtered_widths)

        avg_sizes[cls] = (avg_h, avg_w)

        if avg_h > max_avg_height:
            max_avg_height = avg_h
        if avg_w > max_avg_width:
            max_avg_width = avg_w

    return avg_sizes, (max_avg_height, max_avg_width)


def choise_model(avg_sizes):
    params = {'size':[8,32,64,224], 'model':[SP.SP(), MLP.MLP(), CNN.CNN(), MobileNet.MobileNet()]}
    H, W = avg_sizes
    for idx, res in enumerate(params['size']):
        if H <= res and W <= res :
            end_resolutions = [params['size'][idx:]]
            end_models = [params['model'][idx:]]
            break

    if res <= H and res <= W:
        print('Разрешение больше 224, значит надо начать с малых моделей и разрешения')
        end_resolutions = [params['size'][1], params['size'][2], params['size'][3]]
        end_models = [params['model'][1], params['model'][2], params['model'][3]]
        
    return end_resolutions, end_models

def match_case(params = None):
    match params['resolution']:
        case 8:
            model = SP.SP(num_classes=params['num_classes']).to(params['device'])
        case 32:
            model = MLP.MLP(num_classes=params['num_classes'], hidden_size=params['hidden_size']).to(params['device'])
        case 64:
            model = CNN.CNN(num_classes=params['num_classes'], n_filters = params['n_filters'], dropout_p=params['dropout_p'], hidden_size=params['hidden_size']).to(params['device'])
        case 224:
            model = MobileNet.MobileNet(num_classes=params['num_classes'], alpha=params['alpha']).to(params['device'])
    return model

def save_yaml(name = 'config.yaml', params = None):
    with open(name, 'w') as file:
        yaml.dump(params, file, default_flow_style=False)

def load_yaml(name = 'config.yaml'):
    with open(name, 'r') as file:
        return yaml.safe_load(file)