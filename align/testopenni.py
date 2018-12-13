import cv2
import numpy as np
from primesense import openni2

file_name = b'depth.oni'
openni2.initialize("C:\\Program Files\\OpenNI2\\Redist")
device = openni2.Device
dev = device.open_file(file_name)
depth_stream = dev.create_depth_stream()
depth_stream.start()

while True:
	frame = depth_stream.read_frame()
	frame_data = frame.get_buffer_as_uint16()
	img = np.frombuffer(frame_data, dtype=np.uint16)
	img.shape = (1, 480, 640)
	img = np.concatenate((img, img, img), axis=0)
	img = np.swapaxes(img, 0, 2)
	img = np.swapaxes(img, 0, 1)
	cv2.imshow("image", img)
	cv2.waitKey(34)
openni2.unload()