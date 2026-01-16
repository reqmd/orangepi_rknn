#!/usr/bin/env python3
import socket
import subprocess
import logging

PORT = 4567
BUFFER_SIZE = 1024

# Логирование
# logging.basicConfig(
#     filename="/home/orangepi/udp_server.log",
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(message)s"
# )

# Команды, которые можно выполнять
COMMANDS = {
    "ftp": "/home/ubuntu/NAS-project/scripts/ftp.sh",
    "test": "/home/ubuntu/NAS-project/scripts/test.sh",
    "train": "/home/ubuntu/NAS-project/scripts/train.sh",
    "prep": "/home/ubuntu/NAS-project/scripts/prep.sh"
}

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

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    #logging.info(f"UDP server started on port {PORT}")
    while True:
        data, addr = sock.recvfrom(BUFFER_SIZE)
        message = data.decode("utf-8").strip().lower()
        #logging.info(f"Received '{message}' from {addr}")
        if message in COMMANDS:
            output = run_command(COMMANDS[message])
            response = f"OK: {message} -> {output}"
        else:
            response = "Unknown command"
        #sock.sendto(response.encode("utf-8"), addr)
        sock.sendto(message.encode("utf-8"), addr)

if __name__ == "__main__":
    main()
