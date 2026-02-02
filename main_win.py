import os
import shutil
import subprocess
import stat
import time
import tarfile

from src.data.dataset import LabeledDataset
from src.utils.device_func import device_config
from src.training.train import __train__
from testing.pipeline.test import __test__
from src.training.search_hyperparams import __hyperparams__


params_to_modify = ['configs/static/prep_configs/st_prep_pseudolabel.yaml',       #YAMl файл отвечающий за параметры псевдоразметки
                    'configs/static/prep_configs/st_prep_config.yaml',            #YAMl файл отвечающий за кол-во эпох и устройство, на котором будет проводиться обучение
                    'configs/static/prep_configs/st_prep_hyperparams_config.yaml' #YAMl файл отвечающий за параметры подбора гиперпараметров
                    ]
    
# Изменяет параметр device в зависимости от устройства 
device = device_config(params_roots=params_to_modify) 
print(f'Будет использоваться {device}')

BYTES_TO_COMMAND = {
    1:'rotate',
    2:'new',
    3:'delete',
    4:'copy',
    5:'train',
    6:'test',
    7:'hyperparams'
}

DATA_PATH = './data'
TARS_PATH = './tars'
MODELS_PATH = './models'

def main(mode, arguments):
    mode = BYTES_TO_COMMAND[mode]
    print(f'Mode: {mode}')
    mode_name = arguments[0]
    print(f'ModeName: {mode_name}')
    match mode:
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
                
                #преобразование архива в набор данных
                TARS_PATH = r'C:\Users\Куликов\Desktop\FTP\download'
                if os.listdir(TARS_PATH) == []:
                    print('В папке нет архива\n')
                    return 'В папке нет архива'
                else:
                    archive = os.path.join(TARS_PATH, os.listdir(TARS_PATH)[0])
                    mode_path = os.path.join(DATA_PATH, mode_name, 'images')
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
                        shutil.move(source_item, target_item)
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
            else:
                print('Название режима отсутствует\n')
                return 'Название режима отсутствует'

        case 'delete':
            if mode_name != None:
                if os.path.exists(os.path.join(DATA_PATH, mode_name)):
                    modename_path = os.path.join(DATA_PATH, mode_name)
                    shutil.rmtree(modename_path)
                    print(f'Режим {mode_name} удален')
                else:
                    print('Режим не существует\n')
                    return 'Режим не существует'
            else:
                print('Название режима отсутствует\n')
                return 'Название режима отсутствует'
            
        case 'train':
            if mode_name != None:
                if os.path.exists(os.path.join(DATA_PATH, mode_name)):
                    modename_path = os.path.join(DATA_PATH, mode_name)
                    f_time = time.strftime("%d.%m_%H.%M")
                    model_name = f'{f_time}-{mode_name}.pth'
                    d_path = os.path.join(modename_path, 'images')
                    data = LabeledDataset(d_path)
                    timestamp_train_start = time.time()
                    __train__(data=data, model_name = model_name, modename_path = modename_path)
                    timestamp_train_end = time.time()
                    print(f'Обучение продлилось {timestamp_train_end - timestamp_train_start:.2f} секунд или {(timestamp_train_end - timestamp_train_start) / 60:.2f} минут')
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
                    print(models_list)
                    for model in models_list:
                        model_name = model.split('-')
                        print(model_name[1], f'{arguments[0]}.pth')
                        if len(model_name) < 2:
                            continue
                        if model_name[1] == f'{arguments[0]}.pth':
                            inf_model = os.path.join(MODELS_PATH, model)
                        else:
                            print('Модель для такого режима не найдена\n')
                            
                    
                    TARS_PATH = r'C:\Users\Куликов\Desktop\FTP\download'
                    if os.listdir(TARS_PATH) == []:
                        print('В папке нет архива\n')
                        return 'В папке нет архива'
                    else:
                        archive = os.listdir(TARS_PATH)[0]
                        main_path = os.path.join(DATA_PATH, mode_name)
                        mode_path = os.path.join(DATA_PATH, mode_name, 'test')
                        command = [r'C:\Program Files\7-Zip\7z.exe', "x", os.path.join(TARS_PATH, archive), f"-o{mode_path}", "-y"]
                        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        if os.listdir(mode_path) != [] and result.returncode == 0:
                            print(f'Архив успешно распакован и находится в {mode_path}')
                        else:
                            print('Не удалось распаковать архив или архива нет в нужной папке\n')
                            return 'Не удалось распаковать архив или архива нет в нужной папке'

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
                    print(os.path.join(main_path, 'test'))
                    data = LabeledDataset(os.path.join(main_path, 'test'))
                    __test__(model_name=inf_model, data=data)
                else:
                    print('Режима не существует\n')
                    return 'Режима не существует'
            else:
                print('Название режима отсутствует\n')
                return 'Название режима отсутствует'
                    
        case 'hyperparams':
            if mode_name != None:
                print(os.path.join(DATA_PATH, mode_name))
                if os.path.exists(os.path.join(DATA_PATH, mode_name)):
                    modename_path = os.path.join(DATA_PATH, mode_name)
                    d_path = os.path.join(modename_path, 'images')
                    data = LabeledDataset(d_path)
                    timestamp_train_start = time.time()
                    __hyperparams__(data, mode_name = mode_name, f1_threshhold=0.99)
                    timestamp_train_end = time.time()
                    print(f'Обучение продлилось {timestamp_train_end - timestamp_train_start:.2f} секунд или {(timestamp_train_end - timestamp_train_start) / 60:.2f} минут')
                else:
                    print('Режима не существует\n')
                    return 'Режима не существует'
            else:
                print('Название режима отсутствует\n')
                return 'Название режима отсутствует'

        case _:
            pass
    
while True:
    i = input('Жду команду\n')
    st = i.split(' ')
    if len(st) == 1:
        mode, arguments = int(st[0]), [' ']
    elif len(st) == 3:
        mode, arguments = int(st[0]), [st[2] + st[1]]
    elif len(st) == 5:
        mode, arguments =  int(st[0]), [st[2] + st[1], st[3] + st[4]]
    else:
        print('Unknown command')
    main(mode=mode, arguments=arguments)