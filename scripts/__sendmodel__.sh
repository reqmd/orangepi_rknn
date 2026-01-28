#!/usr/bin/expect -f

set ftp_server "192.168.2.100"
set ftp_user "ubuntu"
set ftp_pass "!q2w3e4R"
set remote_dir "/upload"
set local_dir "/home/ubuntu/NAS-project/models"
set model_name [lindex $argv 0]

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
send "mput $model_name\r"
expect "ftp"
send "bye\r"
interact