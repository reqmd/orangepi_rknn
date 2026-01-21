#!/bin/bash
echo "Получена команда UDP - sendmodel" | socat - udp:192.168.2.1:5000,sp=4568
/home/ubuntu/NAS-project/scripts/__sendmodel__.sh