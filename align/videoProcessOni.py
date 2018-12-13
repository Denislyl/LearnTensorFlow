from scipy import misc
import tensorflow as tf
import detect_face_pb
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from primesense import openni2

ir_file_name = 'ir.avi'
color_file_name = 'color.avi'
depth_file_name = b'depth.oni'
openni2.initialize("C:\\Program Files\\OpenNI2\\Redist")
dev = openni2.Device.open_file(depth_file_name)
depth_stream = dev.create_depth_stream()
depth_stream.start()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

offset = 15
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709

saved_model_dir = "saved_model_dir_signature"

image_path = "./Faces/"
bcheck = os.path.exists(image_path)
if (bcheck is False):
    os.mkdir(image_path)
print('Creating networks and loading parameters')

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face_pb.create_mtcnn(sess, saved_model_dir)


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
capIR = cv2.VideoCapture(ir_file_name)
capColor = cv2.VideoCapture(color_file_name)

# Check if camera opened successfully
if (capIR.isOpened() is False):
    print("Error opening video stream or file")
if (capColor.isOpened() is False):
    print("Error opening video stream or file")

# Read until video is completed
count = 0
while(capIR.isOpened()):
    # Capture frame-by-frame
    ret, frame = capIR.read()
    retColor, frameColor = capColor.read()
    frameDepth = depth_stream.read_frame()
    count += 1
    if ret is True:

        # Display the resulting frame
        # cv2.imshow('Frame', frameColor)
        frame_data = frameDepth.get_buffer_as_uint16()
        imgDepth = np.frombuffer(frame_data, dtype=np.uint16)
        imgDepth.shape = (1, 480, 640)
        imgDepth = np.concatenate((imgDepth, imgDepth, imgDepth), axis=0)
        imgDepth = np.swapaxes(imgDepth, 0, 2)
        imgDepth = np.swapaxes(imgDepth, 0, 1)
        # cv2.imshow("image", imgDepth)
        imgIr = frame
        # cv2.cvtColor(frameColor, cv2.COLOR_BGR2RGB)
        imgColor = frameColor
        # cv2.cvtColor(frameColor, cv2.COLOR_BGR2RGB)
        bounding_boxes, points = detect_face_pb.detect_face(frameColor, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        print('Total face number?{}'.format(nrof_faces))
        crop_faces = []
        for i, face_position in enumerate(bounding_boxes):
            # face_position = face_position.astype(int)
            print(face_position[0:4])
            x1 = int(face_position[0])
            y1 = int(face_position[1])
            x2 = int(face_position[2])
            y2 = int(face_position[3])
            # cv2.rectangle(imgDepth, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(imgColor, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(imgIr, (x1-offset, y1), (x2-offset, y2), (0, 0, 255), 2)
            cropDepth = imgDepth[y1:y2, x1:x2]
            cropImg = imgColor[y1:y2, x1:x2]
            cropIr = imgIr[y1:y2, x1-offset:x2-offset]
        if nrof_faces  > 0:
            save_fileColor = os.path.join(image_path, "Color%s.png"%count)
            save_fileColorOld = os.path.join(image_path, "ColorOld%s.png"%count)
            save_fileIr = os.path.join(image_path, "IR%s.png"%count)
            save_fileIrOld = os.path.join(image_path, "IROld%s.png"%count)
            save_fileDepth = os.path.join(image_path, "Depth%s.png"%count)
            save_fileDepthOld = os.path.join(image_path, "DepthOld%s.png"%count)
            cv2.imwrite(save_fileColor,cropImg)
            cv2.imwrite(save_fileColorOld,imgColor)
            cv2.imwrite(save_fileIr,cropIr)
            cv2.imwrite(save_fileIrOld,imgIr)
            cv2.imwrite(save_fileDepth,cropDepth)
            cv2.imwrite(save_fileDepthOld,imgDepth)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
capIR.release()
capColor.release()
openni2.unload()
# Closes all the frames
cv2.destroyAllWindows()
