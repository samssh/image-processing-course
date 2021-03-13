import cv2 as cv
import numpy as np
import time

if __name__ == '__main__':
    start = time.time()
    file_name = "to_blue.txt"
    file = open(file_name, "r")
    im_name, dest = file.readline().rstrip().split()
    im_GBR = cv.imread(im_name)
    hsv = cv.cvtColor(im_GBR, cv.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    hsv_clone = np.array(hsv, copy=True)
    H_clone = hsv_clone[:, :, 0]
    for line in file.readlines():
        h_min, h_max, s_min, s_max, v_min, v_max, value = [int(a) for a in line.split()]
        condition_h = np.logical_and(h_min <= H, H <= h_max)
        condition_s = np.logical_and(s_min <= S, S <= s_max)
        condition_v = np.logical_and(v_min <= V, V <= v_max)
        condition_line = np.logical_and(np.logical_and(condition_h, condition_s), condition_v)
        H_clone[np.where(condition_line)] += value
    print(time.time() - start)
    cv.imwrite(dest, cv.cvtColor(hsv_clone, cv.COLOR_HSV2BGR))
