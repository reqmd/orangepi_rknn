#!/usr/bin/expect -f

set ftp_server "192.168.2.1"
set ftp_user "ubuntu"
set ftp_pass "!q2w3e4R"
set remote_dir "/download"
set local_dir "/home/ubuntu/NAS-project/model"
set model_name $1

spawn ftp $ftp_server -i
expect "Name"
send "$ftp_user\r"
expect "Password"
send "$ftp_pass\r"
expect "ftp"
send "cd $remote_dir\r"
expect "ftp"
send "lcd $local_dir\r"
expect "ftp"
send "put $1\r"
expect "ftp"
send "bye\r"
interact