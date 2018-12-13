#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = "loki"
import socket
import struct

sendFilePath = "./yuv/yuv_1531563014566.yuv"

while 1:

    # send message
    cmd = input('按回车发送一帧数据 : ').strip()
    if cmd:
        continue

    ipAddr = socket.gethostbyname(socket.gethostname())
    ip_port = ('%s' % ipAddr, 9991)
    stick_pack_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stick_pack_client.connect(ip_port)

    fp = open(sendFilePath, "rb")
    contentBytes = fp.read()
    fp.close()

    size = 60 * 1024
    for i in range(0, len(contentBytes), size):
        stick_pack_client.send(contentBytes[i:i + size])

    data = stick_pack_client.recv(1024)
    stick_pack_client.send(b'\xff\xff\xff\xff\xff\xff\xff\xff')
    stick_pack_client.close()
    print("发送完毕, len = ", len(contentBytes))
