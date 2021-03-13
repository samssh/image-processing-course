import cv2 as cv


def blur_laplacian(im):
    k_size = 9
    blur = cv.GaussianBlur(im, (k_size, k_size), -1)
    laplacian = im - blur
    resized = resize_down(blur)
    return laplacian, resized


def resize_down(im):
    scale = 2
    return cv.resize(im, (int(im.shape[1] / scale), int(im.shape[0] / scale)), interpolation=cv.INTER_CUBIC)


def resize_up(im, shape):
    return cv.resize(im, (shape[1], shape[0]), interpolation=cv.INTER_CUBIC)


def merge(src, dst, mask, k_size):
    k_size = 2 * k_size + 3
    mas = cv.GaussianBlur(mask, (k_size, k_size), -1)
    result = src.copy()
    for i in range(3):
        result[:, :, i] = src[:, :, i] * mas + dst[:, :, i] * (1 - mas)
    return result


def recursive_pyramid(src, dst, mask, depth):
    if depth < end_point:
        src_laplacian, src_resized = blur_laplacian(src)
        dst_laplacian, dst_resized = blur_laplacian(dst)
        result = recursive_pyramid(src_resized, dst_resized, resize_down(mask), depth + 1)
        merge_laplacian = merge(src_laplacian, dst_laplacian, mask, depth)
        result = resize_up(result, merge_laplacian.shape) + merge_laplacian
    else:
        result = merge(src, dst, mask, depth)
    return result


if __name__ == '__main__':
    shark = cv.imread('1.source.jpg').astype(float)
    diver = cv.imread('2.target.jpg').astype(float)
    alpha_c = cv.imread('3.mask.png', 0).astype(float) / 255
    dest_x, dest_y = 0, 560
    h, w, _ = shark.shape
    end_point = 4
    dest = diver[dest_y:dest_y + h, dest_x:dest_x + w, :]
    res = recursive_pyramid(shark, dest, alpha_c, 0)
    res[res > 255] = 255
    res[res < 0] = 0
    diver[dest_y:dest_y + h, dest_x:dest_x + w, :] = res
    diver = diver.astype('uint8')
    cv.imwrite('res2.jpg', diver)
