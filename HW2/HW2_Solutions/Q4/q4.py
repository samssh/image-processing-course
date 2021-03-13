import cv2 as cv
import numpy as np
import numpy.fft as fft


def to_fft(img):
    result = np.empty(img.shape, 'complex')
    for i in range(3):
        ft = fft.fftshift(fft.fft2(img[:, :, i]))
        result[:, :, i] = ft
    return result


def back_fft(img):
    result = np.empty(img.shape, float)
    for i in range(3):
        ft = np.real(fft.ifft2(fft.ifftshift(img[:, :, i])))
        result[:, :, i] = ft
    return result


def gauss_filter(var, h1, w1):
    gauss_h = cv.getGaussianKernel(h1, var)
    gauss_w = cv.getGaussianKernel(w1, var)
    return np.matmul(gauss_h, gauss_w.reshape(1, w1))


def normalize(im):
    res = im - np.min(im)
    if np.max(res) == 0:
        print('fuck')
    res *= 255 / np.max(res)
    return res.astype('uint8')


def three(fi):
    he, we = fi.shape
    res = np.empty((he, we, 3))
    res[:, :, 0] = fi
    res[:, :, 1] = fi
    res[:, :, 2] = fi
    return res


if __name__ == '__main__':
    near = cv.imread('q4_01_near.jpg')
    far = cv.imread('q4_02_far.jpg')

    near_p = np.float32([[670, 330], [635, 654], [1325, 525]])
    far_p = np.float32([[665, 330], [645, 645], [1290, 510]])
    M = cv.getAffineTransform(near_p, far_p)
    h, w, _ = near.shape
    near = cv.warpAffine(near, M, (w, h), borderMode=cv.BORDER_REPLICATE)
    cv.imwrite('q4_03_near.jpg', near)
    cv.imwrite('q4_04_far.jpg', far)

    near_ft = to_fft(near)
    cv.imwrite('q4_05_near.jpg', normalize(np.log(np.abs(near_ft))))

    far_ft = to_fft(far)
    cv.imwrite('q4_06_far.jpg', normalize(np.log(np.abs(far_ft))))

    r = 45
    highpass = gauss_filter(r, h, w)
    highpass /= np.max(highpass)
    highpass = 1 - highpass
    cv.imwrite(f'q4_07_highpass_{r}.jpg', normalize(highpass))

    s = 20
    lowpass = gauss_filter(s, h, w)
    lowpass /= np.max(lowpass)
    cv.imwrite(f'q4_08_lowpass_{s}.jpg', normalize(lowpass))

    a = np.arange(h * w).reshape(h, w)
    dis = np.sqrt((a // w - h // 2) ** 2 + (a % w - w // 2) ** 2)

    r_near = 17
    cutoff_near = np.zeros((h, w))
    cutoff_near[dis > r_near] = 1
    highpass_cutoff = highpass * cutoff_near
    cv.imwrite('q4_09_highpass_cutoff.jpg', normalize(highpass_cutoff))

    r_far = 22
    cutoff_far = np.zeros((h, w))
    cutoff_far[dis < r_far] = 1
    lowpass_cutoff = lowpass * cutoff_far
    cv.imwrite('q4_10_lowpass_cutoff.jpg', normalize(lowpass_cutoff))

    highpassed = three(highpass_cutoff)
    highpassed = highpassed * near_ft
    cv.imwrite('q4_11_highpassed.jpg', normalize(np.log(np.abs(highpassed + 1e-6))))

    lowpassed = three(lowpass_cutoff)
    lowpassed = lowpassed * far_ft
    cv.imwrite('q4_12_lowpassed.jpg', normalize(np.log(np.abs(lowpassed + 1e-6))))

    f1 = np.zeros((h, w))
    f1[dis <= r_near] = 1
    f1 = three(f1)

    f2 = np.zeros((h, w))
    f2[np.logical_and(r_near < dis, dis < r_far)] = 1
    f2 = three(f2)

    f3 = np.zeros((h, w))
    f3[dis >= r_far] = 1
    f3 = three(f3)

    hybrid_frequency = (f1 + 1 * f2) * lowpassed + (f3 + 0.7 * f2) * 3 * highpassed
    cv.imwrite('q4_13_hybrid_frequency.jpg', normalize(np.log(np.abs(hybrid_frequency + 1e-6))))

    hybrid_near = back_fft(hybrid_frequency)
    cv.imwrite('q4_14_hybrid_near.jpg', hybrid_near)

    scale = 10
    hybrid_far = cv.resize(hybrid_near, (hybrid_near.shape[1] // scale, hybrid_near.shape[0] // scale))
    cv.imwrite('q4_15_hybrid_far.jpg', hybrid_far)
