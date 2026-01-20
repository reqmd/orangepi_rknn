#!/bin/bash
echo "Получена команда UDP - ftp" | socat - udp:192.168.2.1:5000,sp=4568
/home/ubuntu/NAS-project/scripts/download.sh
