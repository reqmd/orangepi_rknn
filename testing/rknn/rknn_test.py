import numpy as np
import torch 
from rknn.api import RKNN
import time
from sklearn.metrics import classification_report

from src.utils.config_funcs import load_yaml
from src.utils.save_load import load_model
from src.data.funcs import dataset_into_loader

def export_pytorch_model(params_root, model_name, data):
    model_params = load_yaml(params_root)
    net = load_model(params = model_params, model_name=model_name)
    all_params = load_yaml(params_root)
    res = all_params['resolution']
    net.eval()
    trace_model = torch.jit.trace(net, torch.Tensor(1, 3, res, res))
    rknn_name = './simple_model.pt'
    trace_model.save(rknn_name)
    return rknn_name

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def __rknn__(params_root, model_name, annot_root, data):
    #Загружаем модель и трассируем её
    print('НАЧАЛО ИНФЕРЕНСА')
    timestamp_start_all = timer()
    print('--> Preprocessing')
    rknn_name = export_pytorch_model(params_root=params_root, model_name=model_name)
    all_params = load_yaml(params_root)
    res = all_params['resolution']
    input_size = [[1, 3, res, res]]
    rknn = RKNN(verbose=True)
    print('OK')

    print('--> Config model')
    rknn.config(target_platform='rk3588')
    print('OK')

    print('--> Load model')
    ret = rknn.load_pytorch(model=rknn_name, input_size_list=input_size)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('OK')

    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset=annot_root)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('OK')

    print('--> Export rknn model')
    ret = rknn.export_rknn('./simple_model.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('OK')

    print('--> Creating dataloader with batch_size=1')
    loader = dataset_into_loader(data=data, batch_size=1)

    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('OK')
    print('--> Running model')
    time_loop = []
    y_preds = []
    y_trues = []

    for X, y in loader:
        timestamp_start = timer()
        output = rknn.inference(inputs=X, data_format=['nhwc'])
        timestamp_end = timer()
        time_loop.append(np.round(timestamp_end-timestamp_start, 4))

        y_preds.append(output)
        y_trues.append(y)
    timestamp_end_all = timer()
    print(classification_report(y_trues, y_preds))
    print(f'Обработка изображений заняла {np.sum(time_loop)} секунд, на обработку одного изображения в среднем уходит: {np.mean(time_loop):.4f}')
    print(f'Полный цикл всех действий занял {timestamp_end_all - timestamp_start_all:.4f} секунд')


def timer():
    return time.time()