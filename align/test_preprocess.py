import cv2
import os
import numpy as np

image_path = 'I:/work/images/'
fileName1 = image_path + 'before.png'
fileName2 = "after.png"
depth = cv2.imread(fileName1, cv2.IMREAD_ANYDEPTH)
offset = 150

ret, depth_inv = cv2.threshold(depth, 10 , 10000, cv2.THRESH_BINARY_INV)
depth_out = depth + depth_inv
depth_array = np.asarray(depth_out,  dtype=np.uint16)
# maxLimit = depth_out.min_val + 150
# min_val = depth_out

# depth_array = depth_array - 

save_file = os.path.join(image_path, fileName2)
cv2.imwrite(save_file, depth)
cv2.imshow("image", depth)