import cv2 as cv
import numpy as np
from scipy.sparse import linalg, csc_matrix


def build_indexes(alpha_c):
    ys, xs = np.where(alpha_c > 0)
    points = np.array([ys, xs])
    indexes = np.ones((h, w), dtype=int) * -1
    _, n = points.shape
    for i in range(n):
        indexes[points[0, [i]], points[1, i]] = i
    return indexes, points


def build_sparse_matrix(indexes, points, alpha_c):
    _, n = points.shape
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        y, x = points[0, i], points[1, i]
        if alpha_c[y, x] == 255:
            matrix[i, i] = 4
            matrix[i, indexes[y - 1, x]] = -1
            matrix[i, indexes[y + 1, x]] = -1
            matrix[i, indexes[y, x - 1]] = -1
            matrix[i, indexes[y, x + 1]] = -1
        else:
            matrix[i, i] = 1
    return matrix


def solve_equation(matrix, points, alpha_c, gradiant, src):
    _, n = points.shape
    gradiant[alpha_c < 255] = src[alpha_c < 255]
    b = np.zeros(n, dtype=int)
    for i in range(n):
        y, x = points[0, i], points[1, i]
        b[i] = gradiant[y, x]
    answer = linalg.spsolve(matrix, b)
    answer[answer > 255] = 255
    answer[answer < 0] = 0
    for i in range(n):
        y, x = points[0, i], points[1, i]
        src[y, x] = answer[i]


if __name__ == '__main__':
    shark = cv.imread('1.source.png', cv.IMREAD_UNCHANGED).astype(float)
    dest_x, dest_y = 20, 80
    h, w, _ = shark.shape
    diver = cv.imread("2.target.jpg").astype(float)
    diver_window = diver[dest_y:dest_y + h, dest_x:dest_x + w, :]
    a = shark[:, :, 3]
    shark = shark[:, :, :3]
    gradiant_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    inx, pts = build_indexes(a)
    mat = csc_matrix(build_sparse_matrix(inx, pts, a))
    shark_gradiant = cv.filter2D(shark, -1, gradiant_kernel)
    for ii in range(3):
        solve_equation(mat, pts, a, shark_gradiant[:, :, ii], diver_window[:, :, ii])
    cv.imwrite('res1.jpg', diver.astype('uint8'))
