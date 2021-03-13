import cv2 as cv
import numpy as np

if __name__ == '__main__':
    original = cv.imread("flowers_blur.png").astype('float')
    mask = (cv.GaussianBlur(original, (5, 5), 1) - cv.GaussianBlur(original, (5, 5), 2))
    res = np.empty(mask.shape, float)
    for i in range(3):
        channel = mask[:, :, i]
        h = 15 / np.average(np.abs(channel))
        mask[:, :, i] = channel * h
    res = mask + original
    res[res < 0] = 0
    res[res > 255] = 255
    cv.imwrite("res02.jpg", res.astype('uint8'))
    mask[mask < -127] = -127
    mask[mask > 128] = 128
    mask += 127
    mask = mask.astype("uint8")
    cv.imwrite("res01.jpg", mask)
