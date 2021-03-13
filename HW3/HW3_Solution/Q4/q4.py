import numpy as np
import cv2 as cv
from skimage import segmentation
from skimage import color


def click_event(event, x1, y1, _, params):
    if event == cv.EVENT_LBUTTONDOWN:
        im_show = params[2]
        result = params[3]
        i = params[1][y1][x1]
        result[params[0] == i] = params[4][params[0] == i]
        im_show[params[1] == i] = 0
        cv.imshow('choose birds', im_show)


if __name__ == '__main__':
    im = cv.imread('birds.jpg')
    img = cv.bilateralFilter(im, 2, 150, 150)
    img = cv.medianBlur(img, 3)
    labels = segmentation.felzenszwalb(img, scale=550, min_size=100).astype('uint16')
    img = color.label2rgb(labels, im, bg_label=0)
    img = img * 255
    img = img.astype('uint8')
    h, w, _ = img.shape
    resized_labels = cv.resize(labels, (w // 5, h // 5))
    img = cv.resize(img, (w // 5, h // 5))
    cv.imshow('choose birds', img)
    res = np.zeros(im.shape, 'uint8')
    param = [labels, resized_labels, img, res, im]
    cv.setMouseCallback('choose birds', click_event, param)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite('res08.jpg', res)
