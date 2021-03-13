import cv2 as cv
import numpy as np
import dlib
from imutils import face_utils


def get_index(point, points):
    for (i, p) in enumerate(points):
        if p[0] == point[0] and p[1] == point[1]:
            return i


def get_triangle_by_list(trl, points):
    result = []
    for t in trl:
        pt1 = (int(t[1]), int(t[0]))
        pt2 = (int(t[3]), int(t[2]))
        pt3 = (int(t[5]), int(t[4]))
        result.append((get_index(pt1, points), get_index(pt2, points), get_index(pt3, points)))
    return result


def add_point(sd, ps1, ps2, p):
    sd.insert(p[::-1])
    ps1.append(p)
    ps2.append(p)


def add_4points(sd, ps1, ps2, d):
    add_point(sd, ps1, ps2, (int((height - 1) * d), 0))
    add_point(sd, ps1, ps2, (0, int((width - 1) * d)))
    if d == 1:
        add_point(sd, ps1, ps2, (0, 0))
    else:
        add_point(sd, ps1, ps2, (height - 1, int((width - 1) * d)))
    add_point(sd, ps1, ps2, (int((height - 1) * d), width - 1))


def detect_face(pic):
    gray1 = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)
    rect1 = detector(gray1, 0)[0]
    return face_utils.shape_to_np(predictor(gray1, rect1))


def prepare_triangles(pic1, pic2):
    shape1, shape2 = detect_face(pic1), detect_face(pic2)
    sd = cv.Subdiv2D((0, 0, width, height))
    for (x, y) in shape1:
        sd.insert((int(x), int(y)))
    ps1 = []
    for (x, y) in shape1:
        ps1.append((y, x))
    ps2 = []
    for (x, y) in shape2:
        ps2.append((y, x))
    add_4points(sd, ps1, ps2, 1)
    add_4points(sd, ps1, ps2, 1 / 2)
    add_4points(sd, ps1, ps2, 1 / 4)
    add_4points(sd, ps1, ps2, 3 / 4)
    return get_triangle_by_list(sd.getTriangleList(), ps1), ps1, ps2


def translate_triangle(tra, rect):
    p0 = [tra[0][1] - rect[1], tra[0][0] - rect[0]]
    p1 = [tra[1][1] - rect[1], tra[1][0] - rect[0]]
    p2 = [tra[2][1] - rect[1], tra[2][0] - rect[0]]
    result = np.array([p0, p1, p2], 'float32')
    return result


def wrap_triangle(picture, rect, tra1, tra2):
    tra1 = translate_triangle(tra1, rect)
    tra2 = translate_triangle(tra2, rect)
    affine_matrix = cv.getAffineTransform(tra1, tra2)
    res_size = (rect[2] - rect[0], rect[3] - rect[1])
    picture = cv.warpAffine(picture, affine_matrix, res_size[::-1], borderMode=cv.BORDER_REPLICATE)
    res_mask = np.zeros(res_size, 'uint8')
    pts = np.array([[[tra2[0][0], tra2[0][1]], [tra2[1][0], tra2[1][1]], [tra2[2][0], tra2[2][1]]]], 'int32')
    cv.fillPoly(res_mask, pts, 1)
    res_mask = res_mask.astype(float)
    return picture, res_mask


def get_local_area(tra1, tra2):
    y_min = min(tra1[0][0], tra1[1][0], tra1[2][0], tra2[0][0], tra2[1][0], tra2[2][0])
    x_min = min(tra1[0][1], tra1[1][1], tra1[2][1], tra2[0][1], tra2[1][1], tra2[2][1])
    y_max = max(tra1[0][0], tra1[1][0], tra1[2][0], tra2[0][0], tra2[1][0], tra2[2][0])
    x_max = max(tra1[0][1], tra1[1][1], tra1[2][1], tra2[0][1], tra2[1][1], tra2[2][1])
    return np.array([y_min, x_min, y_max + 1, x_max + 1], int)


def get_frame(picture, points, d_points, trl):
    result = np.zeros((height, width, 3), float)
    mask_result = np.zeros((height, width), float)
    for tri in trl:
        tra1 = np.array([points[tri[0]], points[tri[1]], points[tri[2]]], 'float32')
        tra2 = np.array([d_points[tri[0]], d_points[tri[1]], d_points[tri[2]]], 'float32')
        rect = get_local_area(tra1, tra2)
        wrapped, wrap_mask = wrap_triangle(picture[rect[0]:rect[2], rect[1]:rect[3], :], rect, tra1, tra2)
        wrap_mask = wrap_mask * (1 - mask_result[rect[0]:rect[2], rect[1]:rect[3]])
        mask_result[rect[0]:rect[2], rect[1]:rect[3]] += wrap_mask
        for i in range(3):
            result[rect[0]:rect[2], rect[1]:rect[3], i] += wrapped[:, :, i] * wrap_mask
    return result


if __name__ == '__main__':
    picture1 = cv.imread("p2.jpg")
    picture2 = cv.imread("p1.png")
    path = "shape_predictor_81_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path)
    frame_number = 60
    height, width = picture1.shape[:2]
    triangle_list, points1, points2 = prepare_triangles(picture1, picture2)
    points1 = np.array(points1)
    points2 = np.array(points2)
    writer = cv.VideoWriter('ta.mp4', cv.VideoWriter_fourcc(*"mp4v"), 25, (width, height))
    for fn in range(frame_number + 1):
        time = fn / frame_number
        dest_points = points1 * (1 - time) + points2 * time
        frame1 = get_frame(picture1, points1, dest_points, triangle_list)
        frame2 = get_frame(picture2, points2, dest_points, triangle_list)
        frame = (1 - time) * frame1.astype(float) + time * frame2.astype(float)
        writer.write(frame.astype('uint8'))
    writer.release()
