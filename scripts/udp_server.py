#!/usr/bin/env python3
import socket
import subprocess
import logging
from contextlib import redirect_stderr, redirect_stdout
import sys
from pathlib import Path
import io
 
PORT = 4567
BUFFER_SIZE = 1024
# Настройка логирования для вывода print
print_logger = logging.getLogger('print_logger')
print_logger.setLevel(logging.INFO)
print_file_handler = logging.FileHandler('/home/ubuntu/NAS-project/logs/udp_server_output.log', encoding='utf-8')
print_file_handler.setLevel(logging.INFO)
print_formatter = logging.Formatter('%(asctime)s - %(message)s')
print_file_handler.setFormatter(print_formatter)
print_logger.addHandler(print_file_handler)

# Настройка логирования для ошибок и предупреждений
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.WARNING)
error_file_handler = logging.FileHandler('/home/ubuntu/NAS-project/logs/udp_server_errors.log', encoding='utf-8')
error_file_handler.setLevel(logging.WARNING)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_file_handler.setFormatter(error_formatter)
error_logger.addHandler(error_file_handler)



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
with open ('/home/ubuntu/NAS-project/logs/udp_server_output.log', 'a', encoding='utf-8') as f_out, \
     open ('/home/ubuntu/NAS-project/logs/udp_server_errors.log', 'a', encoding='utf-8') as f_err, \
     redirect_stdout(f_out), \
     redirect_stderr(f_err):
    while True:
        data, addr = sock.recvfrom(BUFFER_SIZE)
        message = data.decode("utf-8").strip().lower().split(' ')
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
        sock.sendto(response.encode("utf-8"), addr)
