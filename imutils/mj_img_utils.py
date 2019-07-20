import cv2
import numpy as np

def open_rgb(image_path):
    img = cv2.imread(str(image_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_rgb_from_zip(zfile, fname):
    data = zfile.read(fname)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


# data = img_zip.read(img_zip.filelist[8])
# img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
# img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)