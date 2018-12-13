import cv2
import numpy as np

class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = self.width * self.height * 3 / 2
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(int(self.frame_len))
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print (str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        return ret, bgr


if __name__ == "__main__":
    #filename = "data/20171214180916RGB.yuv"
    filename = "yuv_160_120_nv21.yuv"
    size = (120, 160)
    cap = VideoCaptureYUV(filename, size)

    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame", frame)
        cv2.waitKey(30)
