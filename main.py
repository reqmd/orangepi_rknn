import os
import tarfile
import shutil
import stat
from datetime import datetime

from testing.rknn.rknn_test import timer
from src.data.dataset import LabeledDataset
from src.utils.device_func import device_config
from src.training.train import __train__
from scripts.run_sh import run_check_call, run_command
from testing.rknn.rknn_test import __rknn__
from logs.logger import mylogger

PRINT_TO_FILE = True
DATA_PATH = './data'
TARS_PATH = './tars'
MODELS_PATH = './models'

COMMANDS = {
    'ftp': "/home/ubuntu/orangepi_rknn/scripts/__ftp__.sh",
    'sendlog':"/home/ubuntu/orangepi_rknn/scripts/__sendlog__.sh",
    'sendannot':"/home/ubuntu/orangepi_rknn/scripts/__sendannot__.sh",
    'sendmodel':"/home/ubuntu/orangepi_rknn/scripts/__sendmodel__.sh",
    'sendresult':'/home/ubuntu/orangepi_rknn/scripts/__sendresult__.sh',
    'ftp_end':'/home/ubuntu/orangepi_rknn/scripts/__ftp_end__.sh'
}

BYTES_TO_COMMAND = {
    0:'testconnect',
    1:'rotate',
    2:'new',
    3:'delete',
    4:'copy',
    5:'train',
    6:'test',
    7:'sendlog',
    8:'sendmodel',
    9:'sendresult',
    10:'stop',
    11:'status',
    12:'ismodeexists'
}

params_to_modify = ['/home/ubuntu/orangepi_rknn/configs/static/prep_configs/st_prep_pseudolabel.yaml',       #YAMl файл отвечающий за параметры псевдоразметки
                    '/home/ubuntu/orangepi_rknn/configs/static/prep_configs/st_prep_config.yaml',            #YAMl файл отвечающий за кол-во эпох и устройство, на котором будет проводиться обучение
                    '/home/ubuntu/orangepi_rknn/configs/static/prep_configs/st_prep_hyperparams_config.yaml' #YAMl файл отвечающий за параметры подбора гиперпараметров
                    ]

    
# Изменяет параметр device в зависимости от устройства 
device = device_config(params_roots=params_to_modify) 
print(f'Будет использоваться {device}')

# Основной цикл
def main(mode, arguments):
    '''Основная функция отвечающая за выбор режима работы программы

    Ключевые аргументы:
    mode -> str - режим работы
    arguments -> list - аргументы для режимов

    Описание режимов:
    testconnect None - проверка подключения к серверу и тестирование модуля main

    rotate None - очистить лог файл

    new [mode_name] - создание нового режима с названием mode_name

    delete [mode_name] - удаление существующего режима mode_name

    copy [old_mode_name, new_mode_name] - копирование режима old_mode_name для получения нового new_mode_name
    
    train [mode_name, ... ] - тренировка модели на наборе данных mode_name с режимом работы train_mode

    test [mode_name, ...] - тестирование модели на наборе данных mode_name 

    Режимы, которые обрабатываются через .sh скрипты
    sendlog None - команда для получения .log файла работы сервера

    sendmodel [mode_name] - команда для получения .pth файла модели после режима test или train

    sendresult [mode_name] - команда для получения результатов работы последнего цикла обучения для режима mode_name
    train будет записывать результат работы в result_train_annot.txt

    Обрабатывается вне main
    stop - Остановка обучения (sudo systemctl restart u.service)
    '''
    LOG_FILE = '/home/ubuntu/orangepi_rknn/logs/udp_server_output.log'
    log = mylogger(LOG_FILE, PRINT_TO_FILE)
    print = log.printml
    
    mode = BYTES_TO_COMMAND[mode]
    print(f'Mode: {mode}')
    mode_name = arguments[0]
    print(f'ModeName: {mode_name}')

    match mode:
        case 'sendresult':
            if mode_name != None:
                annot_root = os.path.join(DATA_PATH, mode_name, 'annotations')
                if os.path.exists(annot_root):
                    result = run_check_call(args=[COMMANDS[mode], annot_root])
                    print(f'chech_call завершился с кодом {result}')
                    if result != 0:
                        return f'Программа завершилась с ошибкой {result}'
                else:
                    print('Папки аннотаций не существует\n')
                    return 'Папки аннотаций не существует'
            else:
                print('Название режима отсутствует\n')
                return 'Название режима отсутствует'
            
        case 'sendlog':
            result = run_command(COMMANDS[mode])
            print(result)

        case 'copy':
            if mode_name != None:
                new_mode_name = arguments[1]
                if new_mode_name != None:
                    shutil.copytree(os.path.join(DATA_PATH, mode_name), os.path.join(DATA_PATH, new_mode_name))
                    print(f'Режим {mode_name} скопирован в {new_mode_name} в {os.path.join(DATA_PATH, new_mode_name)}')
                else:
                    print('Название нового режима отсутствует\n')
                    return 'Название нового режима отсутствует'
            else:
                print('Название режима отсутствует\n')
                return 'Название режима отсутствует'

        case 'new':
            if mode_name != None:
                modename_path = os.path.join(DATA_PATH, mode_name)
                if not os.path.exists(modename_path):
                    os.mkdir(modename_path)
                    os.mkdir(os.path.join(modename_path, 'images'))
                    os.mkdir(os.path.join(modename_path, 'annotations'))
                    os.mkdir(os.path.join(modename_path, 'test'))
                    print(f'Режим {mode_name} был успешно создан')
                else:
                    shutil.rmtree(os.path.join(modename_path, 'images'))
                    shutil.rmtree(os.path.join(modename_path, 'annotations'))
                    shutil.rmtree(os.path.join(modename_path, 'test'))
                    os.mkdir(os.path.join(modename_path, 'images'))
                    os.mkdir(os.path.join(modename_path, 'annotations'))
                    os.mkdir(os.path.join(modename_path, 'test'))
                    print('Режим уже существует\n')
                
                #скачивание архива
                result = run_command(COMMANDS['ftp'])
                print(result)
                
                #преобразование архива в набор данных
                TARS_PATH = './tars'
                if os.listdir(TARS_PATH) == []:
                    print('В папке нет архива\n')
                    return 'В папке нет архива'
                else:
                    archive = os.path.join('tars', os.listdir(TARS_PATH)[0])
                    mode_path = os.path.join(DATA_PATH, mode_name, 'images')
                    #command = ['sudo', 'tar', '-xzvf', archive, '--transform=\'s,\\,/,g\'', '-C', mode_path]
                    #result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    with tarfile.open(archive, "r:gz") as tar:
                        for member in tar.getmembers():
                            # Преобразуем имя файла из CP1251 в UTF-8 (если нужно)
                            try:
                                member.name = member.name.encode('cp1251').decode('utf-8')
                                member.path = member.path.encode('cp1251').decode('utf-8')
                            except UnicodeError:
                                # Если преобразование не удалось, оставляем как есть
                                pass
                            tar.extract(member, path=mode_path)
                        
                    os.chmod(mode_path, 0o755)
                    
                    for root, dirs, files in os.walk(mode_path):
                        for dir in dirs:
                            os.chmod(os.path.join(root, dir), 0o755)  # Права для папок
                        for file in files:
                            os.chmod(os.path.join(root, file), 0o644)  # Права для файлов
                    
                    if os.listdir(mode_path) != []:
                        print(f'Архив успешно распакован и находится в {mode_path}')
                        os.remove(os.path.join(archive))
                    else:
                        print('Не удалось распаковать архив или архива нет в нужной папке\n')
                        return 'Не удалось распаковать архив или архива нет в нужной папке'
                    for item in os.listdir(TARS_PATH):
                        source_item = os.path.join(TARS_PATH, item)
                        target_item = os.path.join(mode_name, item)
                        print(source_item)
                        print(target_item)
                        #shutil.move(source_item, target_item)
                    # преобразование содержимого архива в набор данных
                    classes = os.listdir(mode_path)
                    for cls in classes:
                        cls_path = os.path.join(mode_path, cls)
                        current_mode = os.stat(cls_path).st_mode
                        new_mode = current_mode | stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
                        os.chmod(cls_path, new_mode)
                        cameras_list = os.listdir(cls_path)
                        for camera_num in cameras_list:
                            camera_path = os.path.join(cls_path, camera_num)
                            images_list = os.listdir(camera_path)
                            for image in images_list:
                                os.rename(os.path.join(cls_path, camera_num, image), f'{cls_path}/{camera_num}_{image}')
                            shutil.rmtree(camera_path)
                    
                    # должно получиться class1 - 01_1.bmp, 01_2.bmp, ... 
                    print('Набор данных преобразован в нужный формат')
                    result = run_command(COMMANDS['ftp_end'])
                    print(result)
                    print('Архив удален')

            else:
                print('Название режима отсутствует\n')
                return 'Название режима отсутствует'

        case 'delete':
            if mode_name != None:
                if os.path.exists(os.path.join(DATA_PATH, mode_name)):
                    modename_path = os.path.join(DATA_PATH, mode_name)
                    shutil.rmtree(modename_path)
                    models = os.listdir(MODELS_PATH)
                    for model in models:
                        model_name = model.split('-')[1]
                        if model_name == f'{mode_name}.pth':
                            os.remove(os.path.join(MODELS_PATH, model))
                            print(f'Удалена модель {model}')

                    print(f'Режим {mode_name} удален')
                else:
                    print('Режим не существует\n')
                    return 'Режим не существует'
            else:
                print('Название режима отсутствует\n')
                return 'Название режима отсутствует'

        case 'rotate':
            with open (LOG_FILE, 'w') as f:
                print(f'Лог-файл {LOG_FILE} успешно очищен')

        case 'testconnect':
            print('Проверка на успешное соединение к серверу\n')

        case 'train':
            if mode_name != None:
                if os.path.exists(os.path.join(DATA_PATH, mode_name)):
                    modename_path = os.path.join(DATA_PATH, mode_name)
                    time = datetime.now()
                    f_time = time.strftime("%d.%m_%H.%M")
                    model_name = f'{f_time}-{mode_name}.pth'
                    d_path = os.path.join(modename_path, 'images')
                    data = LabeledDataset(d_path)
                    timestamp_train_start = timer()
                    __train__(data=data, model_name = model_name, modename_path = modename_path)
                    timestamp_train_end = timer()
                    print(f'Обучение продлилось {timestamp_train_end - timestamp_train_start:.2f} секунд или {(timestamp_train_end - timestamp_train_start) / 60:.2f} минут')
                    return 0
                else:
                    print('Режима не существует\n')
                    return 'Режима не существует'
            else:
                print('Название режима отсутствует\n')
                return 'Название режима отсутствует'

        case 'test':
            if mode_name != None:
                if os.path.exists(os.path.join(DATA_PATH, mode_name)):
                    test_path = os.path.join(DATA_PATH, mode_name, 'test')
                    shutil.rmtree(test_path)
                    os.mkdir(test_path)
                    models_list = os.listdir(MODELS_PATH)
                    models_list = sorted(models_list)
                    print(models_list)
                    for model in models_list:
                        model_name = model.split('-')
                        print(model_name)
                        if len(model_name) < 2:
                            continue
                        if model_name[1] == f'{arguments[0]}.pth':
                            inf_model = os.path.join(MODELS_PATH, model)
                            print(f'Выбрана модель {inf_model}')
                        else:
                            print('Модель для такого режима не найдена\n')
                            
                    #скачивание архива
                    result = run_command(COMMANDS['ftp'])
                    print(result)
                    
                    TARS_PATH = './tars'
                    if os.listdir(TARS_PATH) == []:
                        print('В папке нет архива\n')
                        return 'В папке нет архива'
                    else:
                        archive = os.path.join('tars', os.listdir(TARS_PATH)[0])
                        main_path = os.path.join(DATA_PATH, mode_name)
                        mode_path = os.path.join(DATA_PATH, mode_name, 'test')
                        
                        with tarfile.open(archive, "r:gz") as tar:
                          for member in tar.getmembers():
                              # Преобразуем имя файла из CP1251 в UTF-8 (если нужно)
                              try:
                                  member.name = member.name.encode('cp1251').decode('utf-8')
                                  member.path = member.path.encode('cp1251').decode('utf-8')
                              except UnicodeError:
                                  # Если преобразование не удалось, оставляем как есть
                                  pass
                              tar.extract(member, path=mode_path)
                              
                          os.chmod(mode_path, 0o755)
                          
                          for root, dirs, files in os.walk(mode_path):
                            for dir in dirs:
                                os.chmod(os.path.join(root, dir), 0o755)  # Права для папок
                            for file in files:
                                os.chmod(os.path.join(root, file), 0o644)
                        if os.listdir(mode_path) != []:
                            print(f'Архив успешно распакован и находится в {mode_path}')
                            # удаление архива после работы с ним
                            os.remove(archive)
                        else:
                            print('Не удалось распаковать архив или архива нет в нужной папке\n')
                            return 'Не удалось распаковать архив или архива нет в нужной папке'

                    # преобразование содержимого архива в набор данных
                    cameras = os.listdir(mode_path)
                    os.mkdir(os.path.join(mode_path, 'test'))
                    for camera in cameras:
                        camera_path = os.path.join(mode_path, camera)
                        current_mode = os.stat(camera_path).st_mode
                        new_mode = current_mode | stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
                        os.chmod(camera_path, new_mode)
                        images_list = os.listdir(camera_path)
                        for image in images_list:
                            print(os.path.join(camera_path, image))
                            test_path = os.path.join(mode_path, 'test')
                            os.rename(os.path.join(camera_path, image), f'{test_path}/{camera}_{image}')
                        shutil.rmtree(camera_path)
                        
                    result = run_command(COMMANDS['ftp_end'])
                    print(result)
                    print('Архив удален')
                    classes = os.listdir(os.path.join(DATA_PATH, mode_name, 'images'))
                    __rknn__(model_name=inf_model, data_root=main_path, classes = classes)
                    annot_root = os.path.join(DATA_PATH, mode_name, 'annotations')
                    print(f'Путь: {annot_root}')
                    if os.path.exists(annot_root):
                        print(annot_root)
                        result = run_check_call(args=[COMMANDS['sendannot'], annot_root])
                        print(result)
                    else:
                        print('Папки аннотаций не существует\n')
                        return 'Папки аннотаций не существует'
                    
                else:
                    print('Режима не существует\n')
                    return 'Режима не существует'
            else:
                print('Название режима отсутствует\n')
                return 'Название режима отсутствует'
        
        case 'sendmodel':
            if mode_name != None:
                models_list = os.listdir(MODELS_PATH)
                for model in models_list:
                    model_name = model.split('-')
                    print(model_name)
                    if len(model_name) < 2:
                        continue
                    if model_name[1] == f'{mode_name}.pth':
                        mdl = model
                    else:
                        print('Модель для такого режима не найдена\n')
                        return 'Модель для такого режима не найдена'
                if os.path.exists(os.path.join(MODELS_PATH, mdl)):
                    result = run_check_call(args=[COMMANDS[mode], mdl])
                    print(f'chech_call завершился с кодом {result}')
                    if result != 0:
                        return f'Программа завершилась с ошибкой {result}'
                
            else:
                print('Название режима отсутствует\n')
                return 'Название режима отсутствует'
            
        case 'ismodeexists':
            if mode_name != None:
                print(mode_name)
                if not os.path.exists(os.path.join(DATA_PATH, mode_name)):
                    print(f'Режима {mode_name} нет по путю {os.path.join(DATA_PATH, mode_name)}')
                    return f'Режима {mode_name} нет'

            else:
                print('Название режима отсутствует\n')
                return 'Название режима отсутствует'

        case _:
            print(f'Получен неизвестный режим {mode}\n')
            return f'Получен неизвестный режим {mode}'

    print('Конец работы команды\n')
    return 0