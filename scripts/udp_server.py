#!/usr/bin/env python3
import socket
import sys
from pathlib import Path
from datetime import datetime
import traceback
import threading
import subprocess
import struct

PORT = 4567
BUFFER_SIZE = 1024

COMMANDS = {
    'status':'status train mode',
    'stop':'abnormal stop mode',
    'sendlog':"/home/ubuntu/NAS-project/scripts/sendlog.sh",
    'sendannot':"/home/ubuntu/NAS-project/scripts/__sendannot__.sh",
    'sendmodel':"/home/ubuntu/NAS-project/scripts/__sendmodel__.sh",
    'sendresult':'/home/ubuntu/NAS-project/scripts/__sendresult__.sh',
    "copy":"copy mode",
    "new": "new mode",
    "delete": "delete mode",
    "test": "test mode",
    "train": "train mode",
    "rotate": "rotate log mode",
    "testconnect":"test connection with server mode"
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
    8:'sendannot',
    9:'sendmodel',
    10:'sendresult',
    11:'stop',
    12:'status',
}

class mylogger(object):
    def __init__(self, fn='', tofile=False):
        self.fn = fn
        self.tofile = tofile
        return
    def printml(self, *args):
        toprint = ''
        for v in args:
            toprint = toprint + str(v) + ' '
        if self.tofile:
            f = open(self.fn, 'a')
            c_time = datetime.now()
            f_time = c_time.strftime("%m-%d %H:%M:%S.%f")
            f.write(f'{f_time} {toprint}\n')
            f.close
        else: print(toprint)
        return
     
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))

#Логирование принта в файл
LOG_FILE = '/home/ubuntu/NAS-project/logs/udp_server_output.log'
PRINT_TO_FILE = True
log = mylogger(LOG_FILE, PRINT_TO_FILE)
print = log.printml

training_thread = None
stop_event = threading.Event()

def train_model(mode, arguments):
    global result
    print('Вход в поток обучения')
    try:
        result = main.main(mode, arguments)
    except Exception as e:
        print(f"Ошибка в потоке обучения: {e}")
    print('Обучение завершено')


while True:
    data, addr = sock.recvfrom(BUFFER_SIZE)
    print(data)
    if len(data) == 1:
        mode, arguments = BYTES_TO_COMMAND[struct.unpack('B', data[0:1])[0]], [' ']
    elif len(data) == 3:
        mode, arguments = BYTES_TO_COMMAND[struct.unpack('B', data[0:1])[0]], [str(struct.unpack('<H', data[1:3])[0])]
    elif len(data) == 5:
        mode, arguments = BYTES_TO_COMMAND[struct.unpack('B', data[0:1])[0]], [str(struct.unpack('<H', data[1:3])[0]), str(struct.unpack('<H', data[3:5])[0])]
    else:
        message = 'Некоректное число байт'
        sock.sendto(message.encode('utf-8'), addr)
    print(mode, arguments, type(mode), type(arguments))
    try:
        import main
        if mode in COMMANDS:
            if mode == 'status':
                stat = subprocess.run(
                          ["mpstat", "1", "1"],
                          capture_output = True
                      )
                lines = stat.stdout.decode('utf-8').split('\n')
                line = lines[4].split(' ')[-1]
                idle =  float(line)
                if 100 - idle > 75:
                    response = f'OK: {mode} BUSY'
                else:
                    response = f'OK: {mode} IDLE'
            elif mode == 'train':
                # Запускаем обучение в отдельном потоке
                stop_event.clear()
                training_thread = threading.Thread(target=train_model, args=(mode,arguments))
                training_thread.start()
                response = f'OK: {mode}'
            elif mode == 'stop':
                # Останавливаем обучение
                if training_thread and training_thread.is_alive():
                    response = f"OK: {mode} Обучение остановлено"
                    sock.sendto(response.encode("utf-8"), addr)
                    subprocess.run(['sudo', 'systemctl', 'restart', 'u.service'], check = True)
                    stop_event.set()
                    training_thread.join()
                else:
                    response = f"Error: Нет активного процесса обучения"
            else:
                # Выполняем другие команды
                exit_code = main.main(mode, arguments)
                if exit_code != 0:
                    response = f"Error: {mode} {exit_code}"
                else:
                    response = f"OK: {mode}"
        else:
            response = f"Unknown command: {mode}"
    except Exception as e:
        print(str(traceback.format_exc()))
        response = f'Traceback: {mode}'
    sock.sendto(response.encode("utf-8"), addr)
