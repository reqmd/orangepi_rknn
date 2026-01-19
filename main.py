#from logs.logger import logger_info
import sys
import torch
import numpy

from src.utils.device_func import device_config

OS = 'UNKNOWN'
# Определяем операционную систему
if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
    print("Система: Unix (Linux или macOS)")
    OS = 'LINUX'
elif sys.platform.startswith("win"):
    print("Система: Windows")
    OS = 'WIN'
else:
    print(f"Неизвестная система: {sys.platform}")

if OS == 'LINUX':
    params_to_modify = ['/home/ubuntu/NAS-project/configs/static/prep_configs/st_prep_pseudolabel.yaml',       #YAMl файл отвечающий за параметры псевдоразметки
                    '/home/ubuntu/NAS-project/configs/static/prep_configs/st_prep_config.yaml',            #YAMl файл отвечающий за кол-во эпох и устройство, на котором будет проводиться обучение
                    '/home/ubuntu/NAS-project/configs/static/prep_configs/st_prep_hyperparams_config.yaml' #YAMl файл отвечающий за параметры подбора гиперпараметров
                    ]
elif OS == 'WIN':
    params_to_modify = ['configs/static/prep_configs/st_prep_pseudolabel.yaml',       #YAMl файл отвечающий за параметры псевдоразметки
                    'configs/static/prep_configs/st_prep_config.yaml',            #YAMl файл отвечающий за кол-во эпох и устройство, на котором будет проводиться обучение
                    'configs/static/prep_configs/st_prep_hyperparams_config.yaml' #YAMl файл отвечающий за параметры подбора гиперпараметров
                    ]
device_config(params_roots=params_to_modify) #Изменяет параметр device в зависимости от устройства 

def main(mode, arguments = None):
    print('Успешный импорт')
    print(f'Получены следующие аргументы:')
    print(f'Mode: {mode}')
    if arguments != None:
        print(f'Args: {arguments}')
