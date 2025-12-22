#from logs.logger import logger_info
from src.training.search_hyperparams import __hyperparams__
from src.training.pseudo_labeling import __pseudo_labeling__
from testing.pipeline.test import __test__
from src.utils.device_func import device_config

params_to_modify = ['configs/static/prep_configs/st_prep_pseudolabel.yaml', 
                    'configs/static/prep_configs/st_prep_config.yaml', 
                    'configs/static/prep_configs/st_prep_hyperparams_config.yaml']
device_config(params_roots=params_to_modify)










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
# Для пшеницы и ячменя
# root = r'C:\Users\Куликов\Desktop\yapsh'
# test_root = r'C:\Users\Куликов\Desktop\yapsh_test'
# data = LabeledDataset(root)
# num_classes = len(data.classes)
# __hyperparams__(data = data, num_classes=num_classes, test_size=0.5)
# __test__(test_root=test_root, model_name='standart_model.pth')
######################################################