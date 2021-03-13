import cv2 as cv
import numpy as np


def get_by_linear(src, y, x):
    x1 = int(x)
    a = x - x1
    y1 = int(y)
    b = y - y1
    ans = (1 - a) * (1 - b) * src[x1][y1]
    if a != 0:
        ans += a * (1 - b) * src[x1 + 1][y1]
    if b != 0:
        ans += (1 - a) * b * src[x1][y1 + 1]
    if a != 0 and b != 0:
        ans += a * b * src[x1 + 1][y1 + 1]
    return ans


def warp_perspective_on_point(s, src, matrix, w1):
    i, j = s / w1, s % w1
    t, u, _ = np.dot(matrix, [j, i, 1])
    return get_by_linear(src, t, u)


def warp_perspective(src, matrix, d_size):
    result = np.arange(d_size[0] * d_size[1])
    vf = np.vectorize(warp_perspective_on_point, excluded=['src', 'matrix', 'w1'])
    result = vf(result, src=src, matrix=matrix, w1=d_size[1])
    return result.reshape(d_size[0], d_size[1])


def get_book(src_points, image):
    w = int(np.sqrt((src_points[0] - src_points[1]) ** 2).sum())
    h = int(np.sqrt((src_points[0] - src_points[2]) ** 2).sum())
    dst_points = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], 'float32')
    projective_matrix = cv.getPerspectiveTransform(dst_points, src_points)
    img_output = np.empty((h, w, 3), 'uint8')
    for c in range(3):
        img_output[:, :, c] = warp_perspective(image[:, :, c], projective_matrix, (h, w))
    return img_output


if __name__ == '__main__':
    img = cv.imread('books.jpg')
    p00, p10, p01, p11 = [666, 208], [601, 393], [382, 104], [316, 290]
    src_points1 = np.array([p00, p10, p01, p11], 'float32')
    img_output1 = get_book(src_points1, img)
    cv.imwrite('res04.jpg', img_output1)
    p00, p10, p01, p11 = [355, 740], [153, 709], [402, 464], [204, 426]
    src_points2 = np.array([p00, p10, p01, p11], 'float32')
    img_output2 = get_book(src_points2, img)
    cv.imwrite('res05.jpg', img_output2)
    p00, p10, p01, p11 = [814, 968], [610, 1102], [612, 657], [417, 783]
    src_points3 = np.array([p00, p10, p01, p11], 'float32')
    img_output3 = get_book(src_points3, img)
    cv.imwrite('res06.jpg', img_output3)
