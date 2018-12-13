import cv2
import os
import numpy as np
import os

image_path = 'single_face_test_data/'
image_path_positive = 'single_face_test_data/positive_10/'
image_path_after_positive = 'single_face_test_data/positive_after_image/'
image_path_negative = 'single_face_test_data/negative_10/'
image_path_after_negative = 'single_face_test_data/negative_after_image/'

fileName1 = image_path + '1.png'
fileName2 = "after_8to16.png"
fileName3 = image_path_positive + '01_frame_00094_rgb_crop.jpg'





def test_8bit_to_16bit(image_8):
    depth_input = cv2.imread(image_8, cv2.IMREAD_GRAYSCALE)
    depth_array = np.asarray(depth_input,dtype='float32')

    depth_binary = depth_array/255

    offset = 150
    min_val = 400

    for index_x, value_x in enumerate(depth_binary):
        for index_y, value_y in enumerate(value_x):
            if value_y == 1:
                depth_binary[index_x][index_y] = 0
            else:
                depth_binary[index_x][index_y] = (offset * depth_array[index_x][index_y]) + min_val

    image_16 = np.asarray(depth_binary,dtype='uint16')
    return image_16


def list_all_files(rootdir):
    files = []
    file_list = os.listdir(rootdir)
    for i in range(len(file_list)):
        path = os.path.join(rootdir, file_list[i])
        if os.path.isdir(path):
            files.extend(list_all_files(path))
        if os.path.isfile(path):
            files.append(path)
    return files

files = list_all_files(image_path_negative)

for file in files:
    test = test_8bit_to_16bit(file)
    head, tail = os.path.split(file)
    portion = os.path.splitext(tail)
    tail = tail,portion[0]+".png"
    save_file = os.path.join(image_path_after_negative, tail[1])

    cv2.imwrite(save_file, test)
    cv2.imshow("image", test)
    cv2.waitKey(1)


# test = test_8bit_to_16bit(fileName3)
# head, tail = os.path.split(fileName3)
# portion = os.path.splitext(tail)
# tail = tail,portion[0]+".png"
# print(type(tail[1]))



# save_file = os.path.join(image_path_after_positive, tail)

# cv2.imwrite(save_file, test)
# cv2.imshow("image", test)
# cv2.waitKey(1)


