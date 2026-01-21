import numpy as np
import torch
from rknn.api import RKNN
import time
import os
from sklearn.metrics import classification_report

from src.utils.config_funcs import load_yaml
from src.utils.save_load import load_model
from src.data.funcs import dataset_into_loader
from src.data.dataset import LabeledDataset
from logs.logger import mylogger

# логирование в файл результатов обучения
MODELS_PATH = '/home/ubuntu/NAS-project/models'
LOG_FILE = '/home/ubuntu/NAS-project/logs/udp_server_output.log'
PRINT_TO_FILE = True
log = mylogger(LOG_FILE, PRINT_TO_FILE)
print = log.printml

def export_pytorch_model(model_name):
    model_params_root = './configs/dynamic/model_configs/hyperparametrs_search_result_config.yaml'
    model_params = load_yaml(model_params_root)
    net = load_model(params = model_params, model_name=model_name)
    all_params = load_yaml(model_params_root)
    print(all_params)
    res = all_params['resolution']
    net.eval()
    trace_model = torch.jit.trace(net, torch.Tensor(1, 3, res, res))
    rknn_name = './simple_model.pt'
    trace_model.save(rknn_name)
    return rknn_name

def generate_txt(data_root):
    images_root = os.path.join(data_root, 'images')
    folders = os.listdir(images_root) #заведомо должны понимать, что папка должна быть одна
    print(folders)
    if len(folders) != 1:
        print('Папка не одна или её нет, поэтому невозможно сделать тестирование')
        return 1
    else:
        folder = folders[0]
        i_root = os.path.join(images_root, folder)
        images_list = os.listdir(i_root)
        with open(os.path.join(data_root, 'annotations', 'dataset_annot.txt'), 'w') as file:
            for image in images_list:
                file.write(f'{image}\n')
            file.close()
    pass

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def __rknn__(model_name, data_root):
    #mode == None -> просто вывод предсказанных данных
    #mode == 'annot' -> сохраняем метки с уверенностью в классe
    np.set_printoptions(suppress=True, precision=5)
    #Загружаем модель и трассируем её
    print('НАЧАЛО ИНФЕРЕНСА')
    all_params_root = './configs/dynamic/model_configs/hyperparametrs_search_result_config.yaml'
    params = load_yaml(all_params_root)
    timestamp_start_all = timer()
    print('--> Preprocessing')
    rknn_name = export_pytorch_model(model_name=model_name)
    res = params['resolution']
    input_size = [[1, 3, res, res]]
    rknn = RKNN(verbose=True)
    dataset_root = generate_txt(data_root=data_root)
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
    ret = rknn.build(do_quantization=False, dataset=dataset_root)
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
    data = LabeledDataset(os.path.join(data_root, 'images'))
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
    annot_root = os.path.join(data_root, 'annotations', 'result_annot.txt')
    with open(annot_root, 'w') as file:
        file.write('Начало записи аннотаций к разметке\n')
        file.write('Имя файла | Вероятности | Предсказанный класс\n')
        for idx in range(len(data)):
            X, y = data[idx]
            timestamp_start = timer()
            X_array = np.array(X)
            X_array = np.expand_dims(X_array, 0)
            y_raw = rknn.inference(inputs=[X_array], data_format=['nchw'])
            timestamp_end = timer()
            probs = softmax(y_raw[0][0])
            y_pred = np.argmax(probs)
            print(f"Выход модели: {y_raw}, softmax: {probs}, class: {y_pred}")
            print(f"Реальные данные: {y}")
            time_loop.append(np.round(timestamp_end-timestamp_start, 4))
            y_preds.append(y_pred)
            y_trues.append(y)
            file.write(f'{data.image_name}, [{probs[0]:.4f}, {probs[1]:.4f}], {y_pred}\n')

    timestamp_end_all = timer()
    print(classification_report(y_trues, y_preds))
    print(f'Обработка изображений заняла {np.sum(time_loop)} секунд, на обработку одного изображения в среднем уходит: {np.mean(time_loop):.4f}')
    print(f'Полный цикл всех действий занял {timestamp_end_all - timestamp_start_all:.4f} секунд')

def timer():
    return time.time()
