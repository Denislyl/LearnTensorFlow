#!/usr/bin/env python
# -*- coding:utf-8 -*-

import socket
import skeleton_pb2 as pb
import cv2 as cv
import time
import numpy as np
import traceback


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import DataWriter, crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

from SPPE.src.utils.img import im_to_torch
import os
import sys
from tqdm import tqdm
import time
from fn import getTime
import cv2

from pPose_nms import write_json

from detect import identification

args = opt
args.dataset = 'coco'

# TODO: 算法需要处理的地方
# :param isMath 是否需要匹配  false 仅识别骨架; true 识别并匹配
# :param rgbDate RGB数据
# :param width
# :param height
# :returns: float calorie       卡路里
# :returns: float score         得分（相当于当前帧的匹配度）
# :returns: string skeleton     当前帧的骨架
#def identification(isMath, rgbDate, width, height, det_model, pose_model):

    # 模拟识别过程 ...
    #time.sleep(0.05)

    #return 3.2, 56, str("xxxxxxxxxxxxxxxxxxx")


# 返回给Client的数据包
# :param: int command       命令字（SkeletonReq.Command）
# :param: long time         时间戳
# :param: float calorie     卡路里
# :param: float score       得分（相当于当前帧的匹配度）
# :param: string skeleton   当前帧的骨架
# :returns: bytes           返回给Client的二进制数据
def packagingData(command, time, calorie, score, skeleton):
    res = pb.SkeletonRes()
    res.cammand = command
    res.time = time
    res.calorie = calorie
    res.score = score
    res.skeleton = skeleton

    contentBytes = res.SerializeToString()
    # 组装包头
    contentSize = len(contentBytes)
    resBytes = bytearray()
    resBytes += b'\xff\xff\xff\xff\xff\xff\xff\xff'
    sizeBytes = contentSize.to_bytes(4, byteorder="big", signed=False)
    for k in range(0, 4):
        resBytes[k + 4] = sizeBytes[k];
    resBytes += contentBytes
    return resBytes

# :param dataBytes bytes 收到Client的请求
# :returns: bytes        处理请求后返回给Client
def unpackingData(dataBytes, det_model, pose_model):


    ######## 解析请求数据
    req = pb.SkeletonReq()
    req.ParseFromString(dataBytes)

    if req.cammand == pb.SkeletonReq.DETECTION:
        print("<< 收到请求 : 骨架检测 DETECTION", ", time = ", req.time, ", size(", req.width, " * ", req.height, "), len = ", len(req.data))
    else:
        print("<< 收到请求 : 识别 MATCH", ", time = ", req.time, ", size(", req.width, " * ", req.height, "), len = ", len(req.data))

    # if 1 == 1:
       # return packagingData(command=req.cammand, time=req.time, calorie=324.2, score=435, skeleton="xxxxxxdfefddxxxxxx")

    # yuv >> RGB
    height = req.width
    width = req.height
    shape = (int(height * 3 / 2), width)
    yuv = np.frombuffer(req.data, dtype=np.uint8)
    #yuv = yuv.reshape(shape)
    #rgb = cv.cvtColor(yuv, cv.COLOR_YUV2RGB_NV21)  # COLOR_YUV2RGB_NV21  COLOR_YUV2BGR_NV21
    rgb = req.data
    # rgb to bmp 
    # numpy.ndarray
    # print("type(rgb) = ", type(rgb))

    # 写入文件
    fileName = './rgb/rgb_' + str(req.time) + '.jpg'
    cv.imwrite(fileName, rgb, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    #fp = open(fileName, 'wb+')
    #fp.write(rgb)
    #fp.close()

    # 识别骨架和匹配，耗时操作，其他线程，不阻塞
    rgb = cv2.imread(fileName)
    indentiRes = identification(det_model, pose_model, isMatch = (req.cammand == pb.SkeletonReq.MATCH), rgbDate = rgb, width=width, height=height)

    # TODO: 判断结果队列是否有消息：
    # 有则返回消息，无则返回 0xFF0xFF0xFF0xFF

    # 将待返回的数据打包
    resPackageData = packagingData(command=req.cammand, time=req.time, calorie=indentiRes[0], score=indentiRes[1], skeleton=indentiRes[2])

    print("indentiRes = ", indentiRes)
    print("回复 Client >> ", len(resPackageData), " >> ",  resPackageData)

    return resPackageData

def startServer():

    ipAddr = socket.gethostbyname(socket.gethostname())

    buff_size = 60 * 1000
    ip_port = ('%s' % ipAddr, 9991)

    stick_pack_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stick_pack_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    stick_pack_server.bind(ip_port)
    stick_pack_server.listen(5)
    
    # Load YOLO model
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
    det_model.load_weights('models/yolo/yolov3-spp.weights')
    det_model.net_info['height'] = args.inp_dim
    det_inp_dim = int(det_model.net_info['height'])
    assert det_inp_dim % 32 == 0
    assert det_inp_dim > 32
    det_model.cuda()
    det_model.eval()
    
    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()
    
    while 1:
        print("\n >>>  local ip : %s " % ipAddr, " Waiting port : ", 9991)
        client, address = stick_pack_server.accept()
        client.settimeout(10)
        print("accept Client : ", client)

        start = (int(round(time.time() * 1000)))  # 毫秒级时间戳
        frameIndex = 0

        buffSize = 0
        cash_buff = bytearray()
        while 1:
            try:
                cmd = client.recv(buff_size)  # bytes
                if not client:
                    break
                if len(cmd) <= 0:
                    time.sleep(0.005)
                if len(cmd) > 0:
                    # print(cmd.encode('hex'))

                    if str(cmd[:8].hex()).lower() == "ffffffffffffffff":
                        print("Client 主动关闭啦...")
                        end = (int(round(time.time() * 1000)))  # 毫秒级时间戳
                        print("fps = ", int(1000 * frameIndex / (end - start)))

                        client.close()
                        break

                    if str(cmd[:4].hex()).lower() == "ffffffff":
                        buffSize = int(cmd[4:8].hex(), 16)
                        cash_buff = bytearray()
                    if len(cash_buff) > 0:
                        cash_buff += cmd
                    else:
                        cash_buff += cmd[8:]
                    # print("recv >> ", len(cmd), buffSize, len(cash_buff))
                    if len(cash_buff) == buffSize:
                        frameIndex += 1
                        resBytes = unpackingData(cash_buff, det_model, pose_model)
                        client.send(resBytes)

                        # 将数据原包返回
                        # size = 60 * 1024
                        # for i in range(0, len(cash_buff), size):
                        #     client.send(cash_buff[i:i + size])
                    continue
            except Exception as e:
                traceback.print_exc()
                print('---->', e)
                break
        client.close()
        print("close one Client ...")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

if __name__ == '__main__':
    startServer()