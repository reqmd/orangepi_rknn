#!/usr/bin/env python3
import socket
import sys
from pathlib import Path
from datetime import datetime
import traceback
 
PORT = 4567
BUFFER_SIZE = 1024

COMMANDS = {
    'ftp': "/home/ubuntu/NAS-project/scripts/ftp.sh",
    'sendlog':"/home/ubuntu/NAS-project/scripts/sendlog.sh",
    'sendannot':"/home/ubuntu/NAS-project/scripts/sendannot.sh",
    'sendmodel':"/home/ubuntu/NAS-project/scripts/sendmodel.sh",
    "new": "new mode",
    "delete": "delete mode",
    "test": "test mode",
    "train": "train mode",
    "rotate": "rotate log mode",
    "raiseerr":"raise error mode",
    "extract":"extract archive mode",
    "testconnect":"test connection with server mode"
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

while True:
    data, addr = sock.recvfrom(BUFFER_SIZE)
    message = data.decode("utf-8").strip().lower().split(' ')
    if len(message) > 1:
        mode, arguments = message[0], message[1:]
    else:
        mode, arguments = message[0], [' ']
    print(mode, arguments)
    try:
        import main
        if mode in COMMANDS:
            exit_code = main.main(mode, arguments)
            if exit_code != 0:
                response = f"Error: {message} {exit_code}"
            else:
                response = f"OK: {message}"
        else:
            response = "Unknown command"
    except Exception as e:
        print(str(traceback.format_exc()))
        response = 'Программа закончилась с какой то ошибкой, смотреть лог'
    sock.sendto(response.encode("utf-8"), addr)
