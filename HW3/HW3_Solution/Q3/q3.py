import numpy as np
import cv2 as cv
import time


def center_parameters(center_numbers, h1, w1):
    a1 = int(np.sqrt(center_numbers * h1 / w1) + 0.5)
    b1 = int(center_numbers / a1 + 0.5)
    s1 = int(np.sqrt(h1 * w1 / (a1 * b1)))
    return a1, b1, s1


def center_coordinates(a1, b1, s1, h1, w1):
    k1 = a1 * b1
    centers1 = np.empty((k1, 2))
    centers1[:, 0] = (np.linspace(0, k1 - 1, k1) // b1) * s1 + ((h1 - s1 * (a1 - 1)) // 2)
    centers1[:, 1] = (np.linspace(0, k1 - 1, k1) % b1) * s1 + ((w1 - s1 * (b1 - 1)) // 2)
    return centers1.astype(int)


def program(k, alpha, dest):
    start = time.time()
    im = cv.imread("slic.jpg")
    gray = cv.cvtColor(im, cv.COLOR_BGRA2GRAY)
    grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    grad = cv.magnitude(grad_x, grad_y)
    h, w, _ = im.shape
    a, b, s = center_parameters(k, h, w)
    centers = center_coordinates(a, b, s, h, w)
    k = a * b
    labels = np.ones((h, w), 'uint16')

    lab_image = cv.cvtColor(im, cv.COLOR_BGR2LAB).astype(float)
    feature_vex = np.empty((h, w, 5))
    feature_vex[:, :, 0] = lab_image[:, :, 0]
    feature_vex[:, :, 1] = lab_image[:, :, 1]
    feature_vex[:, :, 2] = lab_image[:, :, 2]
    feature_vex[:, :, 3] = (np.linspace(0, h * w - 1, h * w, dtype=int) % w).reshape(h, w)
    feature_vex[:, :, 4] = (np.linspace(0, h * w - 1, h * w, dtype=int) // w).reshape(h, w)
    print(k)
    for i in range(k):
        ns = 2  # 5 // 2 = 2
        y = centers[i, 0]
        x = centers[i, 1]
        grad_n = grad[y - ns:y + ns + 1, x - ns:x + ns + 1]
        t = np.where(grad_n == np.min(grad_n))
        centers[i, :] = y + t[0][0] - ns, x + t[1][0] - ns
        ss = s // 2
        sp = s - ss
        y_min = max(y - ss, 0)
        if y_min < ss:
            y_min = 0
        y_max = min(y + sp, h)
        if h - y_max < ss:
            y_max = h
        x_min = max(x - ss, 0)
        if x_min < ss:
            x_min = 0
        x_max = min(x + sp, w)
        if w - x_max < ss:
            x_max = w
        labels[y_min:y_max, x_min:x_max] = i
    distances = np.zeros((h, w), float)
    for _ in range(10):
        distances[:, :] = 1e9
        for i in range(k):
            y = centers[i, 0]
            x = centers[i, 1]
            y_min = max(y - s, 0)
            y_max = min(y + s, h)
            x_min = max(x - s, 0)
            x_max = min(x + s, w)
            color_dis = (feature_vex[y_min:y_max, x_min:x_max, 0] - feature_vex[y, x, 0]) ** 2
            color_dis += (feature_vex[y_min:y_max, x_min:x_max, 1] - feature_vex[y, x, 1]) ** 2
            color_dis += (feature_vex[y_min:y_max, x_min:x_max, 2] - feature_vex[y, x, 2]) ** 2
            color_dis = np.sqrt(color_dis)
            xy_dis = (feature_vex[y_min:y_max, x_min:x_max, 3] - feature_vex[y, x, 3]) ** 2
            xy_dis += (feature_vex[y_min:y_max, x_min:x_max, 4] - feature_vex[y, x, 4]) ** 2
            xy_dis = np.sqrt(xy_dis)
            dis = color_dis + alpha * xy_dis
            accepted = dis <= distances[y_min:y_max, x_min:x_max]
            labels[y_min:y_max, x_min:x_max][accepted] = i
            distances[y_min:y_max, x_min:x_max][accepted] = dis[accepted]
        for i in range(k):
            y = centers[i, 0]
            x = centers[i, 1]
            y_max = min(y + s, h)
            y_min = max(y - s, 0)
            x_min = max(x - s, 0)
            x_max = min(x + s, w)
            yi, xi = np.where(labels[y_min:y_max, x_min:x_max] == i)
            if yi.shape[0] == 0:
                continue
            y_av, x_av = np.average(yi), np.average(xi)
            centers[i, :] = [y_av + y_min, x_av + x_min]
    for _ in range(20):
        labels = cv.medianBlur(np.int16(labels), 5)
    l_grad_x = cv.Sobel(labels, cv.CV_64F, 1, 0, ksize=1)
    l_grad_y = cv.Sobel(labels, cv.CV_64F, 0, 1, ksize=1)
    l_grad = cv.magnitude(l_grad_x, l_grad_y)
    im[l_grad > 0] = [0, 0, 0]
    cv.imwrite(dest, im)
    print(time.time() - start)


if __name__ == '__main__':
    program(64, 0.5, 'res05.jpg')
    program(256, 1, 'res06.jpg')
    program(1024, 2, 'res07.jpg')
    program(2048, 4.8, 'res08.jpg')
