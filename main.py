#from logs.logger import logger_info
import os

from src.training.search_hyperparams import __hyperparams__
from src.training.pseudo_labeling import __pseudo_labeling__
from src.training.train import __train__
from src.utils.device_func import device_config
from src.data.dataset import LabeledDataset
from src.data.funcs import create_annot, replace_new_ftp_data, replace_annot_to_ftp
from src.utils.run_bash import run_bash
from testing.pipeline.test import __test__
#from testing.rknn.rknn_test import __rknn__, timer

from torchvision import transforms

params_to_modify = ['configs/static/prep_configs/st_prep_pseudolabel.yaml',       #YAMl файл отвечающий за параметры псевдоразметки
                    'configs/static/prep_configs/st_prep_config.yaml',            #YAMl файл отвечающий за кол-во эпох и устройство, на котором будет проводиться обучение
                    'configs/static/prep_configs/st_prep_hyperparams_config.yaml' #YAMl файл отвечающий за параметры подбора гиперпараметров
                    ]
device_config(params_roots=params_to_modify) #Изменяет параметр device в зависимости от устройства 

def main(mode, arguments = None):
    print('Успешный импорт')
    return mode, arguments
####################################################

# Цикл обучения
#data = LabeledDataset(root)
#timestamp_train_start = timer()
#__train__(data=data)
#timestamp_train_end = timer()
#print(f'Обучение продлилось {timestamp_train_end - timestamp_train_start:.2f} секунд или {(timestamp_train_end - timestamp_train_start) / 60:.2f} минут')

# Цикл тестирования
#transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
#test_data = LabeledDataset(test_root, test=True, transform=transform)
#__test__(test_root=test_root, model_name='standart_model.pth')
#create_annot(data=test_data, txt_root=dataset_root)

# Цикл инференса на устройстве
#__rknn__(model_name='standart_model.pth', annot_root=annot_root, dataset_root=dataset_root, data=test_data, mode = 'images')
#replace_annot_to_ftp(src_root=annot_root_without_file, dst_root=ftp_root)

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
