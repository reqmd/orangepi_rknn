#!/bin/bash
echo "Получена команда UDP - train" | socat - udp:192.168.2.1:5000,sp=4568
exit 0
