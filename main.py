import sys
import os

from src.utils.device_func import device_config
from logs.logger import mylogger

LOG_FILE = '/home/ubuntu/NAS-project/logs/udp_server_output.log'
PRINT_TO_FILE = True
OS = 'UNKNOWN'
DATA_PATH = './data'

#Логирование принта в файл
log = mylogger(LOG_FILE, PRINT_TO_FILE)
print = log.printml

# Определяет операционную систему
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
    
# Изменяет параметр device в зависимости от устройства 
device = device_config(params_roots=params_to_modify) 
print(f'Будет использоваться {device}')

# Основной цикл
def main(mode, arguments = None):
    print('Успешный импорт')
    print(f'Получены следующие аргументы:')
    print(f'Mode: {mode}')
    mode_name = arguments[0]
    print(f'ModeName: {mode_name}')
    print(f'Other Args: {arguments[1:]}')
    match mode:
        case 'new':
            if mode_name != None:
                if not os.path.exists(os.path.join(DATA_PATH, mode_name)):
                    modename_path = os.path.join(DATA_PATH, mode_name)
                    os.mkdir(modename_path)
                    os.mkdir(os.path.join(modename_path, 'images'))
                    os.mkdir(os.path.join(modename_path, 'annotations'))
                    print(f'Режим {mode} был успешно создан')
                else:
                    print('Режим уже существует, воспользуйтесь delete и создайте режим заново')
            else:
                print('Название режима отсутствует')

        case 'delete':
            if mode_name != None:
                if not os.path.exists(os.path.join(DATA_PATH, mode_name)):
                    modename_path = os.path.join(DATA_PATH, mode_name)
                    os.system(f"sudo rm -rf {modename_path}")
                else:
                    print('Режим не существует')
            else:
                print('Название режима отсутствует')

        case 'rotate':
            with open (LOG_FILE, 'w') as f:
                print(f'Лог-файл {LOG_FILE} успешно очищен')

        case 'openarch':
            pass

        case _:
            print(f'Получен неизвестный режим {mode}')

    print('\n')
