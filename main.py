import sys
import os
import shutil
import subprocess

from src.utils.device_func import device_config
from logs.logger import mylogger

LOG_FILE = '/home/ubuntu/NAS-project/logs/udp_server_output.log'
PRINT_TO_FILE = True
OS = 'UNKNOWN'
DATA_PATH = './data'
TARS_PATH = './tars'


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
                if os.path.exists(os.path.join(DATA_PATH, mode_name)):
                    modename_path = os.path.join(DATA_PATH, mode_name)
                    shutil.rmtree(modename_path)
                    print(f'Режим {mode_name} удален')
                else:
                    print('Режим не существует')
            else:
                print('Название режима отсутствует')

        case 'rotate':
            with open (LOG_FILE, 'w') as f:
                print(f'Лог-файл {LOG_FILE} успешно очищен')

        case 'extract':
            if mode_name != None:
                if os.listdir(TARS_PATH) == []:
                    print('В папке нет архива')
                else:
                    archive = os.listdir(TARS_PATH)[0]
                    mode_path = os.path.join(DATA_PATH, mode_name, 'images')
                    command = ["/usr/bin/7z", "x", os.path.join(TARS_PATH, archive), f"-o{mode_path}", "-y"]
                    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    #запись работы 7z
                    print(result.stdout.decode("utf-8"))
                    print(result.stderr.decode("utf-8"))
                    if os.listdir(mode_path) != [] and result.returncode == 0:
                        print(f'Архив успешно распакован и находится в {mode_path}')
                    else:
                        print('Не удалось распаковать архив или архива нет в нужной папке')
            else:
                print('Название режима отсутствует')

        case 'testconnect':
            print('Проверка на успешное соединение к серверу')

        case _:
            print(f'Получен неизвестный режим {mode}')

    print('\n')
