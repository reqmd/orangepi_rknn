import numpy as np

#from logs.logger import logger_info
from src.training.search_hyperparams import __hyperparams__
from src.training.pseudo_labeling import __pseudo_labeling__
from src.training.train import __train__
from src.utils.device_func import device_config
from src.data.dataset import LabeledDataset
from src.data.funcs import create_annot, replace_new_ftp_data
from testing.pipeline.test import __test__
from testing.rknn.rknn_test import __rknn__, timer

from torchvision import transforms

params_to_modify = ['configs/static/prep_configs/st_prep_pseudolabel.yaml', 
                    'configs/static/prep_configs/st_prep_config.yaml', 
                    'configs/static/prep_configs/st_prep_hyperparams_config.yaml']
params_root = 'configs/static/prep_configs/st_prep_hyperparams_config.yaml'
device_config(params_roots=params_to_modify)

ftp_root = './ftp'
dst_root = './data/from_ftp'
replace_new_ftp_data(ftp_root, dst_root)

####################################################
# Для пшеницы и ячменя Linux
root = './data/data/yapsh'
test_root = './data/data/yapsh_test'
txt_root = './data/annotations/yapsh_test/dataset.txt'

data = LabeledDataset(root)
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
test_data = LabeledDataset(test_root, test=True, transform=transform)

# timestamp_train_start = timer()
# __train__(data=data)
# timestamp_train_end = timer()
# print(f'Обучение продлилось {timestamp_train_end - timestamp_train_start:.2f} секунд или {(timestamp_train_end - timestamp_train_start) / 60:.2f} минут')
create_annot(data=test_data, txt_root=txt_root)
__rknn__(params_root=params_root, model_name='standart_model.pth', annot_root=txt_root, data=test_data)
__test__(test_root=test_root, model_name='standart_model.pth')
######################################################

##########################################################
# #Для риса
# root = r'C:\Users\Куликов\rice_dataset\train_valid'
# new_data_root = r'C:\Users\Куликов\rice_dataset\not_stratified_data'
# test_root = r'C:\Users\Куликов\rice_dataset\test'
# data = LabeledDataset(root)
# num_classes = len(data.classes)
# #logger_info(__hyperparams__, 'logs/hyperparams.log', data = data, num_classes=num_classes) #с логированием
# __hyperparams__(data = data, num_classes=num_classes)
# __pseudo_labeling__(data, test_root)
# print('Проверка метрик до псведоразметки')
# __test__(test_root=new_data_root, model_name='standart_model.pth')
# print('Проверка метрик после псведоразметки')
# __test__(test_root=test_root, model_name='new_model.pth')
##############################################################

####################################################
# Для пшеницы и ячменя Windows
# root = r'C:\Users\Куликов\Desktop\yapsh'
# test_root = r'C:\Users\Куликов\Desktop\yapsh_test'
# data = LabeledDataset(root)
# num_classes = len(data.classes)
# __hyperparams__(data = data, num_classes=num_classes, test_size=0.5)
# __test__(test_root=test_root, model_name='standart_model.pth')
######################################################
