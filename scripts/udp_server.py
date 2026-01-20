#!/usr/bin/env python3
import socket
import subprocess
import sys
from pathlib import Path
from datetime import datetime

 
PORT = 4567
BUFFER_SIZE = 1024

COMMANDS = {
    'ftp': "/home/ubuntu/NAS-project/scripts/ftp.sh",
    'sendlog':"/home/ubuntu/NAS-project/scripts/log.sh",
    "new": "new mode",
    "delete": "delete mode",
    "test": "test mode",
    "train": "train mode",
    "rotate": "rotate log mode",
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

def run_command(cmd):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            check=True,
            text=True
        )
        return result.stdout or "OK"
    except Exception as e:
        return f"ERROR: {e}"

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
            if mode != ('ftp' and 'sendlog'):
                main.main(mode, arguments)
            else:
                output = run_command(COMMANDS[mode])
                print(output)
            response = f"OK: {message}"
        else:
            response = "Unknown command"
    except ImportError as e:
        print(f'Ошибка импорта: {e}')
    sock.sendto(response.encode("utf-8"), addr)
