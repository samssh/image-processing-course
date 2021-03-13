import numpy as np
import cv2 as cv
import sklearn.cluster as cl

if __name__ == '__main__':
    im = cv.imread('park.jpg')
    h, w, _ = im.shape
    img = cv.resize(im, (w // 8, h // 8), interpolation=cv.INTER_AREA)
    h, w, _ = img.shape
    pixels = np.empty((h * w, 5), float)
    pixels[:, :3] = img.reshape(h * w, 3) / 255
    pixels[:, 3] = 0.3 * (np.linspace(0, h * w - 1, h * w, dtype=int) // w) / h
    pixels[:, 4] = 0.3 * (np.linspace(0, h * w - 1, h * w, dtype=int) % w) / w
    cluster = cl.MeanShift(bandwidth=0.1, bin_seeding=True)
    t = cluster.fit(pixels)
    ac = t.labels_.reshape((h, w))
    h, w, _ = im.shape
    ac = cv.resize(ac.astype('uint8'), (w, h), interpolation=cv.INTER_NEAREST)
    ac = cv.medianBlur(ac, 9)
    im = im.astype(float)
    n = ac.max() + 1
    for i in range(n):
        for j in range(3):
            tt = np.average((im[:, :, j])[ac == i])
            (im[:, :, j])[ac == i] = int(tt)
    cv.imwrite('res04.jpg', im.astype('uint8'))
