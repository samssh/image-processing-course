import cv2 as cv
import numpy as np
import math
import time


def get_diff(channel1, channel2, max_translate):
    c1 = channel1[max_translate:-1 * max_translate, max_translate:-1 * max_translate]
    c2 = channel2[max_translate:-1 * max_translate, max_translate:-1 * max_translate]
    k = ((c1 - c2) ** 2).sum()
    return math.sqrt(k)


def check(channel1, channel2, max_translate, base_x, base_y):
    res, x, y = 0, 0, 0
    for i in range(-1 * max_translate, max_translate):
        for j in range(-1 * max_translate, max_translate):
            t = np.float32([[1, 0, i + base_x], [0, 1, j + base_y]])
            h, w = channel2.shape[:2]
            img_translation = cv.warpAffine(channel2, t, (w, h))
            if i == -1 * max_translate and j == -1 * max_translate:
                res = get_diff(channel1, img_translation, max_translate + max(base_y, base_x))
                x, y = i, j
            else:
                r = get_diff(channel1, img_translation, max_translate + max(base_y, base_x))
                if r < res:
                    res = r
                    x, y = i, j
    return x + base_x, y + base_y


def find_translate(image):
    b, g, r = three_part(image)
    if image.shape[0] < 500 and image.shape[1] < 500:
        g_x, g_y = check(b, g, 20, 0, 0)
        r_x, r_y = check(b, r, 20, 0, 0)
        return g_x, g_y, r_x, r_y
    else:
        resized = cv.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)), interpolation=cv.INTER_AREA)
        g_x, g_y, r_x, r_y = find_translate(resized)
        g_x, g_y = check(b, g, 7, 2 * g_x, 2 * g_y)
        r_x, r_y = check(b, r, 7, 2 * r_x, 2 * r_y)
        return g_x, g_y, r_x, r_y


def three_part(mat):
    he = int(mat.shape[0] / 3)
    channel1 = mat[:he, :]
    channel2 = mat[he:2 * he, :]
    channel3 = mat[2 * he:3 * he, :]
    return channel1, channel2, channel3


if __name__ == '__main__':
    ti = time.time()
    im = cv.imread("D:/uni/image processing/HW/HW1/Questions/melons.tif", -1)
    kernel = np.array(([[-1], [1]]), np.float32)
    mask = cv.filter2D(im.astype(float), -1, kernel) + 2 ** 16
    _, threshes = cv.threshold(mask, 2 ** 16, 2 ** 16 - 1, cv.THRESH_BINARY)
    tr_g_x, tr_g_y, tr_r_x, tr_r_y = find_translate(threshes.astype('float64'))
    print(tr_g_x, tr_g_y, tr_r_x, tr_r_y)
    b_channel, g_channel, r_channel = three_part(im)
    height, width = b_channel.shape[:2]
    tr_mat = np.float32([[1, 0, tr_g_x], [0, 1, tr_g_y]])
    g_channel = cv.warpAffine(g_channel, tr_mat, (width, height))
    tr_mat = np.float32([[1, 0, tr_r_x], [0, 1, tr_r_y]])
    r_channel = cv.warpAffine(r_channel, tr_mat, (width, height))
    result = np.zeros((height, width, 3), np.uint8)
    result[:, :, 0] = (b_channel / 256).astype('uint8')
    result[:, :, 1] = (g_channel / 256).astype('uint8')
    result[:, :, 2] = (r_channel / 256).astype('uint8')
    cv.imwrite("res04.jpg", result)
    print(time.time() - ti)
