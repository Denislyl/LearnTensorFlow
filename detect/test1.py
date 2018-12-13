import cv2
import os
import numpy as np
from PIL import Image

image_path = 'I:/work/images/'
fileName1 = image_path + '01_frame_00172_rgb_crop.jpg'
fileName2 = "after1.png"
depth = cv2.imread(fileName1, cv2.IMREAD_GRAYSCALE)

depth_array = np.asarray(depth ,dtype='uint16')

save_file = os.path.join(image_path, fileName2)
cv2.imwrite(save_file, depth_array)
cv2.imshow("image", depth_array)