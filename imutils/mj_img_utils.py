import cv2
import numpy as np

def open_rgb(image_path):
    img = cv2.imread(str(image_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_rgb_from_zip(zfile, fname):
    data = zfile.read(fname)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# data = img_zip.read(img_zip.filelist[8])
# img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
# img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)