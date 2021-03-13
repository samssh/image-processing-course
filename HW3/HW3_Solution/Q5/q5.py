import cv2 as cv
import numpy as np
import collections
import time


def click_handler(event, x1, y1, _, params):
    if event == cv.EVENT_LBUTTONDOWN:
        params[1][0], params[1][1] = y1, x1
        params[0][0], params[0][1] = y1, x1
    if event == cv.EVENT_LBUTTONUP:
        draw_line(params[1][1], params[1][0], x1, y1, params[3], params[2])
        cv.destroyAllWindows()
    if event == cv.EVENT_MOUSEMOVE and first_point[0] != -1:
        draw_line(params[0][1], params[0][0], x1, y1, params[3], params[2])
        params[0][0], params[0][1] = y1, x1
        cv.imshow('draw a curve', showing_image.astype('uint8'))


def draw_line(start_x, start_y, end_x, end_y, pl, im_show):
    a = np.linspace(0, 1, max(np.abs(start_y - end_y), np.abs(start_x - end_x)) // 2)
    for al in a:
        ta = 1 - al
        add_point(int(start_x * ta + end_x * al), int(start_y * ta + end_y * al), pl, im_show)


def add_point(x1, y1, pl, im_show):
    if len(pl) == 0:
        pl.append((y1, x1))
        im_show[y1, x1] = [255, 255, 255]
    else:
        if (pl[-1][0] - y1) ** 2 + (pl[-1][1] - x1) ** 2 > 250:
            pl.append((y1, x1))
            im_show[y1, x1] = [255, 255, 255]


def draw_on_image(img, point_array):
    temp = np.copy(img)
    for ii in range(pn):
        s_x, s_y, e_x, e_y = point_array[ii - 1][1], point_array[ii - 1][0], point_array[ii][1], point_array[ii][0]
        temp = cv.line(temp, (e_x, e_y), (s_x, s_y), [255, 0, 0])
        temp = cv.circle(temp, (e_x, e_y), 2, [0, 0, 255], thickness=-1)
    return temp


def preprocess(img):
    result = cv.GaussianBlur(img, (15, 15), -1)
    result = cv.Canny(result, 30, 40)
    result = result.astype(int)
    return result


if __name__ == '__main__':
    last_point = [-1, -1]
    first_point = [-1, -1]
    image = cv.imread('tasbih.jpg').astype(int)
    showing_image = np.copy(image)
    h, w, _ = showing_image.shape
    point_list = collections.deque()
    cv.imshow('draw a curve', showing_image.astype('uint8'))
    param = [last_point, first_point, showing_image, point_list]
    cv.setMouseCallback("draw a curve", click_handler, param)
    cv.waitKey(0)
    start = time.time()
    writer = cv.VideoWriter('contour.mp4', cv.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    pn = len(point_list)
    points = np.empty((pn, 2), int)
    print(pn)
    for i in range(pn):
        points[i, 0] = point_list[i][0]
        points[i, 1] = point_list[i][1]
    writer.write(draw_on_image(image, points).astype('uint8'))
    grad_image = preprocess(image.astype('uint8'))
    t = 2
    m = 2 * t + 1
    x_shift = (np.linspace(0, m ** 2 - 1, m ** 2).reshape((m, m))) % m
    y_shift = (np.linspace(0, m ** 2 - 1, m ** 2).reshape((m, m))) // m
    d1 = 1000
    itn = 0
    while d1 > 0:
        energy = np.zeros((pn, m, m), float)
        last_detected = np.empty((pn, m, m, 2), int)
        d = 0
        for i in range(pn):
            d += np.sqrt(((points[i, :] - points[i - 1, :]) ** 2).sum())
        beta = 500
        gama = 5000
        if itn < 50:
            alpha = 0.9
        else:
            alpha = 1
        d *= alpha / pn
        y_m = np.average(points[:, 0])
        x_m = np.average(points[:, 1])
        for i in range(pn):
            for x in range(m):
                for y in range(m):
                    x_now, y_now = x + points[i][1] - t, y + points[i][0] - t
                    y2 = (y_shift + points[i - 1, 0] - (y + points[i][0])) ** 2
                    x2 = (x_shift + points[i - 1, 1] - (x + points[i][1])) ** 2
                    ll = (1 + itn / 20) * ((y2 + x2 - d) ** 2) + energy[i - 1, :, :]
                    mc = np.where(ll == np.min(ll))
                    energy[i][y][x] = ll[mc[0][0], mc[1][0]]
                    gradient = grad_image[y + points[i][0] - t, x + points[i][1] - t]
                    energy[i][y][x] -= gama * gradient
                    energy[i][y][x] += beta * ((y_now - y_m) ** 2 + (x_now - x_m) ** 2) / (gradient * 100 + 1)
                    last_detected[i][y][x][0], last_detected[i][y][x][1] = mc[0][0], mc[1][0]
        last_y = np.where(energy[pn - 1, :, :] == np.min(energy[pn - 1, :, :]))[0][0]
        last_x = np.where(energy[pn - 1, :, :] == np.min(energy[pn - 1, :, :]))[1][0]
        points[pn - 1, 0] += last_y - t
        points[pn - 1, 1] += last_x - t
        d1 = 0
        for i in range(pn - 1, 0, -1):
            last_y, last_x = last_detected[i][last_y][last_x]
            points[i - 1, 0] += last_y - t
            points[i - 1, 1] += last_x - t
            d1 += (last_y - t) ** 2 + (last_x - t) ** 2
        writer.write(draw_on_image(image, points).astype('uint8'))
        itn += 1
        print(itn)
    writer.release()
    cv.imwrite('res09.jpg', draw_on_image(image, points).astype('uint8'))
    print(time.time() - start)
