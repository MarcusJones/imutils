import cv2
import numpy as np
import matplotlib.patches as patches
import math
import logging

def open_rgb(image_path):
    img = cv2.imread(str(image_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_rgb_from_zip(zfile, fname):
    data = zfile.read(fname)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    raise
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


def rle_to_pixels(rle_code):
    '''
    Transforms a RLE code string into a list of pixels of a (768, 768) canvas
    '''
    rle_code = [int(i) for i in rle_code.split()]
    pixels = [(pixel_position % 768, pixel_position // 768)
                 for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2]))
                 for pixel_position in range(start, start + length)]
    return pixels


def convert_rle_mask(mask, shape=(768,768)):
    # convert RLE mask into 2d pixel array
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    s = mask.split()
    for i in range(len(s)//2):
        start = int(s[2*i]) - 1
        length = int(s[2*i+1])
        img[start:start+length] = 1
    img2 = img.reshape(shape).T
    logging.debug("Converted {} entries from Run Length Encoding to mask pixels ndarray image".format(i))
    return img2


def get_bbox(img):
    # get bounding box for a mask
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def get_bbox_vertices(img):
    x1, x2, y1, y2 = get_bbox_p(img)
    return (x1, y1), (x1, y2), (x2, y2), (x2, y1)

def get_bbox_polygon(img):
    points = get_bbox_vertices(img)

def get_bbox_p(img, padding=5):
    # add padding to the bounding box
    x1,x2,y1,y2 = get_bbox(img)
    lx,ly = img.shape
    x1 = max(x1-padding,0)
    x2 = min(x2+padding+1, lx-1)
    y1 = max(y1-padding,0)
    y2 = min(y2+padding+1, ly-1)
    return x1,x2,y1,y2


def convert_box(box):
    # convert parameters of the box for plotting
    rot1 = math.cos(box[4])
    rot2 = math.sin(box[4])
    bx1 = box[0] - 0.5*(box[2]*rot1 - box[3]*rot2)
    bx2 = box[1] - 0.5*(box[2]*rot2 + box[3]*rot1)
    return (bx1,bx2,box[2],box[3],box[4]*180.0/math.pi)


def get_rec(box,width=1):
    b = convert_box(box)
    return patches.Rectangle((b[0],b[1]),b[2],b[3],b[4],linewidth=width,edgecolor='g',facecolor='none')


def get_contour(mask):
    """Return a cv2 contour object from a binary 0/1 mask"""
    assert mask.ndim == 2
    assert mask.min() == 0
    assert mask.max() == 1
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    logging.debug("Returning {} fit contours over mask pixels".format(len(contours)))
    return contour


def fit_draw_rect(img, contour):
    """"""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img2 = cv2.drawContours(img,[box],0,(0,0,255),2)
    # plt.imshow(img2)
    # plt.show()


def fit_draw_ellipse(img, contour, thickness=2):
    # returns the rotated rectangle in which the ellipse is inscribed
    rotated_rect = cv2.fitEllipse(contour)
    logging.debug("Fit ellipse drawing to a rotated rectangle at {}".format(rotated_rect[0]))
    # (center x, center y), (width, height), angle
    # Draw the ellipse object into the image
    # Return the new image
    img2 = cv2.ellipse(img=img, box=rotated_rect, color=(0,255,0), thickness=thickness)
    return img2


def fit_draw_axes_lines(img, contour, thickness=2):
    # (x1,y1), (x2,y2), angle = cv2.fitEllipse(contour)

    rect = cv2.minAreaRect(contour)
    vertices = cv2.boxPoints(rect)

    # cv2.minEllipse[element].points(cv2.fitEllipse(contour))
    img = cv2.line(img, tuple((vertices[0] + vertices[1])/2), tuple((vertices[2] + vertices[3])/2), (0,255,0), thickness=thickness)
    img = cv2.line(img, tuple((vertices[1] + vertices[2])/2), tuple((vertices[3] + vertices[0])/2), (0,255,0), thickness=thickness)

    # pt1,pt2,_ = ellipse
    # img4 = cv2.line(img3, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    logging.debug("Fit and draw 2 lines to major/minor axis of rotated rect".format())
    return img


def draw_ellipse_and_axis(img, contour, thickness=1):
    img = fit_draw_ellipse(img, contour, thickness)
    img = fit_draw_axes_lines(img, contour, thickness)
    return img
