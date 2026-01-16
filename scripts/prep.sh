#!/bin/bash
echo "Получена команда UDP - prep" | socat - udp:192.168.2.1:5000,sp:4568

