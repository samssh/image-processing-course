import cv2 as cv
import numpy as np


def calculate(k, template, reference, t1):
    i1 = k // t1
    j1 = k % t1
    ht, wt = template.shape
    refs = normalize(reference[i1:i1 + ht, j1:j1 + wt])
    return (refs * template).sum()


def normalize(signal):
    normal_signal = signal - np.average(signal)
    var_signal = np.sqrt(np.sum(normal_signal ** 2))
    normal_signal /= var_signal
    return normal_signal


if __name__ == '__main__':
    pat_original = cv.imread('patch.png')
    pat = cv.cvtColor(pat_original, cv.COLOR_BGR2GRAY).astype(float)
    original = cv.imread('Greek_ship.jpg')
    im = cv.cvtColor(original, cv.COLOR_BGR2GRAY).astype(float)
    scale = 10
    tem = cv.resize(pat, (pat.shape[1] // scale, pat.shape[0] // scale))
    ref = cv.resize(im, (im.shape[1] // scale, im.shape[0] // scale))
    tem = normalize(tem)
    h1, w1 = ref.shape
    h2, w2 = tem.shape
    h, w = h1 - h2, w1 - w2
    x = range(h * w)
    v_cal = np.vectorize(calculate, excluded=['template', 'reference', 't1'])
    result = v_cal(x, template=tem, reference=ref, t1=w)
    result = result.reshape(h, w)
    result = np.abs(result)
    maxi = 210 * np.max(result) / 255
    result[result < maxi] = 0
    result[result >= maxi] = 255
    scale2 = 3
    result = cv.resize(result, (result.shape[1] // scale2, result.shape[0] // scale2), interpolation=cv.INTER_AREA)
    lists = []
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if result[i][j] > 0:
                u = []
                for s in range(len(lists)):
                    t = lists[s]
                    dis = (t[0] - i) ** 2 + (t[1] - j) ** 2
                    if dis < 50:
                        u.append(s)
                if len(u) == 0:
                    lists.append([i, j, result[i, j]])
                else:
                    for s in u:
                        if lists[s][2] < result[i][j]:
                            lists[s] = [i, j, result[i][j]]
    for s in range(len(lists)):
        x, y, _ = lists[s]
        x *= scale2 * scale
        y *= scale2 * scale
        pt1 = (y, x)
        pt2 = (y + pat_original.shape[1], x + pat_original.shape[0])
        original = cv.rectangle(original, pt1, pt2, (255, 0, 0), 5)
    cv.imwrite('res03.jpg', original.astype('uint8'))
