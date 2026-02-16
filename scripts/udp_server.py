#!/usr/bin/env python3
import socket
import sys
import time
from pathlib import Path
from datetime import datetime
import traceback
import threading
import subprocess
import struct

PORT = 5001
BUFFER_SIZE = 1024

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

RESPONSE_TO_BYTES = {
    0:'Ok',
    1:'Error',
    2:'Unknown command',
    3:'Traceback'
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
message = ' '

def sendfivetimes(byte_data):
    for i in range(1, 6):
        sock.sendto(byte_data, addr)
        print(f"Отправлен пакет {i} клиенту {addr}")
        time.sleep(0.05)

def train_model(mode, arguments):
    global result
    print('Вход в поток обучения')
    try:
      resp = 0
      result = main.main(mode, arguments)
      print('Обучение завершено')
      if result == 0:
          resp = 4
          response = [resp, 5]
          byte_data = bytes(response)
          print(byte_data)
          sendfivetimes(byte_data=byte_data)
          print(f'Успешно выполнена команда {mode}')
      else:
          resp = 1
          response = [resp, 5]
          byte_data = bytes(response)
          print(byte_data)
          sendfivetimes(byte_data=byte_data)
          print(f'Команда {mode} была выполнена с ошибкой')
    except Exception as e:
      print(str(traceback.format_exc()))
      resp = 3
      response = [resp, 5]
      byte_data = bytes(response)
      print(byte_data)
      sendfivetimes(byte_data=byte_data)

while True:
    data, addr = sock.recvfrom(BUFFER_SIZE)
    print(data)
    if len(data) == 1:
        mode, arguments = struct.unpack('B', data[0:1])[0], [' ']
    elif len(data) == 3:
        mode, arguments = struct.unpack('B', data[0:1])[0], [str(struct.unpack('<H', data[1:3])[0])]
    elif len(data) == 5:
        mode, arguments = struct.unpack('B', data[0:1])[0], [str(struct.unpack('<H', data[1:3])[0]), str(struct.unpack('<H', data[3:5])[0])]
    # else:
    #     message = 'Некоректное число байт'
    #     sock.sendto(message.encode('utf-8'), addr)
    print(mode, arguments, type(mode), type(arguments))
    resp = 0
    try:
        import main
        if mode in BYTES_TO_COMMAND:
            if mode == 11:
                if training_thread is not None and training_thread.is_alive():
                    response = [resp, mode, 1]
                    byte_data = bytes(response)
                    print(byte_data)
                    print('Train')
                    sock.sendto(byte_data, addr)
                else:
                    response = [resp, mode, 0]
                    byte_data = bytes(response)
                    print(byte_data)
                    print('Idle')
                    sock.sendto(byte_data, addr)
            elif mode == 5:
                # Запускаем обучение в отдельном потоке
                response = [resp, mode]
                byte_data = bytes(response)
                print(byte_data)
                #sendfivetimes(byte_data=byte_data)
                sock.sendto(byte_data, addr)
                stop_event.clear()
                training_thread = threading.Thread(target=train_model, args=(mode,arguments))
                training_thread.start()
                #print(response)
            elif mode == 10:
                # Останавливаем обучение
                if training_thread and training_thread.is_alive():
                    response = [resp, mode]
                    byte_data = bytes(response)
                    print(byte_data)
                    #print(response)
                    sock.sendto(byte_data, addr)
                    subprocess.run(['sudo', 'systemctl', 'restart', 'u.service'], check = True)
                    stop_event.set()
                    training_thread.join()
                else:
                    resp = 1
                    response = [resp, mode]
                    byte_data = bytes(response)
                    print(byte_data)
                    #print(response)
            elif (mode == 2) or (mode == 6):
                response = [resp, mode]
                byte_data = bytes(response)
                sock.sendto(byte_data, addr)
                #sendfivetimes(byte_data=byte_data)
                exit_code = main.main(mode, arguments)
                if exit_code != 0:
                    response = [1, mode]
                    byte_data = bytes(response)
                    #print(response)
                    sendfivetimes(byte_data=byte_data)
                else:
                    response = [4, mode]
                    byte_data = bytes(response)
                    print(byte_data)
                    sendfivetimes(byte_data=byte_data)
                    #print(response)
            else:
                # Выполняем другие команды
                exit_code = main.main(mode, arguments)
                if exit_code != 0:
                    resp = 1
                    response = [resp, mode]
                    byte_data = bytes(response)
                    print(byte_data)
                    #print(response)
                    sock.sendto(byte_data, addr)
                    message = exit_code
                    print(f'Сообщение ошибки {message}')
                else:
                    response = [resp, mode]
                    byte_data = bytes(response)
                    print(byte_data)
                    sock.sendto(byte_data, addr)
                    #print(response)
        else:
            resp = 2
            response = [resp, mode]
            byte_data = bytes(response)
            print(byte_data)
            sock.sendto(byte_data, addr)
            #print(response)
    except Exception as e:
        print(str(traceback.format_exc()))
        resp = 3
        response = [resp, mode]
        byte_data = bytes(response)
        sock.sendto(byte_data, addr)
        print(byte_data)
        #print(response)
