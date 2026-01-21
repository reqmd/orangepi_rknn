#!/bin/bash
echo "Получена команда UDP - sendlog" | socat - udp:192.168.2.1:5000,sp=4567
/home/ubuntu/NAS-project/scripts/__sendlog__.sh
