#!/usr/bin/env python3
import socket
import subprocess
import logging
import sys
from pathlib import Path
 
PORT = 4567
BUFFER_SIZE = 1024

# Логирование
# logging.basicConfig(
#     filename="/home/orangepi/udp_server.log",
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(message)s"
# )

COMMANDS = {
    "ftp": "/home/ubuntu/NAS-project/scripts/ftp.sh",
    "test": "test mode",
    "train": "train mode",
    "prep": "preprocessing mode"
}

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
#logging.info(f"UDP server started on port {PORT}")
while True:
    data, addr = sock.recvfrom(BUFFER_SIZE)
    message = data.decode("utf-8").strip().lower().split(' ')
    #logging.info(f"Received '{message}' from {addr}")
    mode, arguments = message[0], message[1:]
    print(mode)
    try:
        print(f"Добавлен путь: {project_root}")
        print(f"Текущие пути: {sys.path}")
        import main
        if mode in COMMANDS:
            if mode != 'ftp':
                main.main(mode, arguments)
                output = 'OK'
            else:
                output = run_command(COMMANDS[mode])
            response = f"OK: {message} -> {output}"
        else:
            response = "Unknown command"
    except ImportError as e:
        print(f'Ошибка импорта: {e}')
    #sock.sendto(message[0].encode("utf-8"), addr)
    sock.sendto(response.encode("utf-8"), addr)
