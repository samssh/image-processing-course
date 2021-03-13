import cv2 as cv
import numpy as np


def find_cut_x(dif):
    dif = dif[:, gauss_kernel:ovs]
    dif = dif[:, :, 0] + dif[:, :, 1] + dif[:, :, 2]
    cost = np.zeros((bs, ovs - gauss_kernel), int)
    detected = np.zeros((bs - 1, ovs - gauss_kernel), int)
    cost[:, 0] = dif[:, 0]  # set for first row
    for yy in range(bs - 1):
        for xx in range(ovs - gauss_kernel):
            # update yy+1, xx
            minimum = xx
            if 0 < xx and dif[yy, xx - 1] < dif[yy, minimum]:
                minimum = xx - 1
            if xx < ovs - gauss_kernel - 1 and dif[yy, xx + 1] < dif[yy, minimum]:
                minimum = xx + 1
            detected[yy, xx] = minimum
            cost[yy + 1, xx] = dif[yy + 1][xx] + cost[yy, minimum]
    h = np.where(cost[bs - 1, :] == cost[bs - 1, :].min())[0]
    detected_x = h[np.random.randint(0, h.size)]
    result = np.zeros((bs, bs), bool)
    for i in range(bs - 1, -1, -1):
        result[i, 0:detected_x + gauss_kernel] = True
        detected_x = detected[i - 1, detected_x]
    return result


def find_cut_y(dif):
    dif = dif[gauss_kernel:ovs, :]
    dif = dif[:, :, 0] + dif[:, :, 1] + dif[:, :, 2]
    detected = np.zeros((ovs - gauss_kernel, bs - 1), int)
    cost = np.zeros((ovs - gauss_kernel, bs), int)
    cost[:, 0] = dif[:, 0]  # set for first row
    for xx in range(bs - 1):
        for yy in range(ovs - gauss_kernel):
            # update yy, xx + 1
            minimum = yy
            if 0 < yy and dif[yy - 1, xx] < dif[minimum, xx]:
                minimum = yy - 1
            if yy < ovs - gauss_kernel - 1 and dif[yy + 1, xx] < dif[minimum, xx]:
                minimum = yy + 1
            detected[yy, xx] = minimum
            cost[yy, xx + 1] = dif[yy][xx + 1] + cost[minimum, xx]
    h = np.where(cost[:, bs - 1] == cost[:, bs - 1].min())[0]
    detected_y = h[np.random.randint(0, h.size)]
    result = np.zeros((bs, bs), bool)
    for i in range(bs - 1, -1, -1):
        result[0:detected_y + gauss_kernel, i] = True
        detected_y = detected[detected_y, i - 1]
    return result


def find_cut(b, mb):
    temp = np.zeros((bs, bs), bool)
    d = (b - mb) ** 2
    if x != 0:
        temp = np.logical_or(temp, find_cut_x(d))
    if y != 0:
        temp = np.logical_or(temp, find_cut_y(d))
    result = np.zeros((bs, bs), float)
    result[temp] = 1
    return result


def prepare_matched_block(b, mb):
    b1 = np.copy(mb)
    n = np.sum(target_mask[y:y + bs, x:x + bs])
    for i in range(3):
        avd = np.sum(b[:, :, i]) / n - np.average(b1[:, :, i])
        avd /= 2
        if avd > 0:
            m = b1 > 255 - avd
            b1 = b1 + avd
            b1[m] = 255
        else:
            m = b1 < -avd
            b1 = b1 + avd
            b1[m] = 0
    return b1


def match(y1, x1):
    if x1 == 0 and y1 == 0:
        return np.random.randint(0, texture.shape[0] - bs), np.random.randint(0, texture.shape[1] - bs)
    template = target[y1:y1 + bs, x1:x1 + bs]
    mask1 = target_mask[y1:y1 + bs, x1:x1 + bs]
    method = cv.TM_CCORR_NORMED
    result = cv.matchTemplate(texture, template, method, mask=mask1)
    result = np.abs(result)
    u = ([], [])
    mini = result.min()
    for _ in range(100):
        _, _, _, max_loc = cv.minMaxLoc(result)
        u[0].append(max_loc[1])
        u[1].append(max_loc[0])
        result[max_loc[1], max_loc[0]] = mini - 1
    i = np.random.randint(0, len(u[0]))
    return u[0][i], u[1][i]


if __name__ == '__main__':
    bs = 150  # block size
    ovs = 30  # overlap size
    target_size = 2550
    gauss_kernel = 5
    steps = int((target_size - bs) / (bs - ovs)) + 1
    texture = cv.imread('texture1.jpg')
    texture = cv.medianBlur(texture, 5)
    target = np.zeros((target_size, target_size, 3), 'uint8')
    target_mask = np.zeros((target_size, target_size), 'uint8')
    for y in range(0, target_size - bs + 1, bs - ovs):
        for x in range(0, target_size - bs + 1, bs - ovs):
            y_match, x_match = match(y, x)
            # apply
            if x == 0 and y == 0:
                target[y:y + bs, x:x + bs] = texture[y_match:y_match + bs, x_match:x_match + bs]
                target_mask[y:y + bs, x:x + bs] = 1
                continue
            block = target[y:y + bs, x:x + bs]
            matched_block = prepare_matched_block(block, texture[y_match:y_match + bs, x_match:x_match + bs])
            mask = find_cut(block, matched_block)
            mask = cv.GaussianBlur(mask, (gauss_kernel, gauss_kernel), -1)
            for ii in range(3):
                target[y:y + bs, x:x + bs, ii] = matched_block[:, :, ii] * (1 - mask) + block[:, :, ii] * mask
            target_mask[y:y + bs, x:x + bs] = 1
    cv.imwrite('res1.jpg', target)
