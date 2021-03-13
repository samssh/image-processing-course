import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time

if __name__ == '__main__':
    start = time.time()
    dark = cv.imread('IMG_20201105_014914.jpg')
    pink = cv.imread('res06.jpg')
    result = np.empty(dark.shape, 'uint8')
    color = ('b', 'g', 'r')
    for i in range(3):
        his_dark = cv.calcHist([dark], [i], None, [256], [0, 256])
        his_pink = cv.calcHist([pink], [i], None, [256], [0, 256])
        cdf_dark = his_dark.cumsum()
        cdf_dark /= cdf_dark[255]
        cdf_pink = his_pink.cumsum()
        cdf_pink /= cdf_pink[255]
        t = np.zeros(256, int)
        for j in range(256):
            t[j] = np.count_nonzero(cdf_pink <= cdf_dark[j]) - 1
        result[:, :, i] = t[dark[:, :, i]]
        hist = cv.calcHist([result], [i], None, [256], [0, 256])
        plt.plot(hist / hist.sum(), color=color[i])
        plt.xlim([0, 256])
    cv.imwrite("sam.jpg", result)
    # plt.savefig("res05.jpg")
    print(time.time() - start)
