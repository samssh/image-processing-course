import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def show_data(data, labels, centers, plot):
    a = data[labels.ravel() == 0]
    b = data[labels.ravel() == 1]
    plot.scatter(a[:, 0], a[:, 1], c='b', s=5)
    plot.scatter(b[:, 0], b[:, 1], c='r', s=5)
    plot.scatter(centers[:, 0], centers[:, 1], s=40, c='y', marker='s')
    show_ax(plot)


def show_ax(plot):
    plot.grid(True, which='both')
    plot.axhline(y=0, color='k')
    plot.axvline(x=0, color='k')


def set_label(axs, x_label, y_label):
    for ax in axs.flat:
        ax.set(xlabel=x_label, ylabel=y_label)
        ax.label_outer()


if __name__ == '__main__':
    file = open('Points.txt', 'r')
    n = int(file.readline())
    xy = np.empty((n, 2), 'float32')
    for i in range(n):
        xp, yp = [float(a) for a in file.readline().split()]
        xy[i][0] = xp
        xy[i][1] = yp

    figure, axes = plt.subplots(1, 2)

    axes[0].set_title("Cartesian coordinate system")
    axes[0].scatter(xy[:, 0], xy[:, 1], s=5)
    show_ax(axes[0])
    axes.flat[0].set(xlabel='x', ylabel='y')

    tet = np.arctan(xy[:, 1] / xy[:, 0])
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    rt = np.empty((n, 2), 'float32')
    rt[:, 0] = r
    rt[:, 1] = tet

    axes[1].set_title("Polar coordinate system")
    axes[1].scatter(r, tet, s=5)
    show_ax(axes[1])
    axes.flat[1].set(xlabel='r', ylabel='\u03B8')

    figure.set_figheight(5)
    figure.set_figwidth(10)
    figure.suptitle("data in two coordinate system")
    figure.savefig('res01.jpg')

    figure, axes = plt.subplots(1, 2)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 0.1)
    _, label, center = cv.kmeans(xy, 2, None, criteria, 1000, cv.KMEANS_RANDOM_CENTERS)
    show_data(xy, label, center, axes[0])
    axes[0].set_title('itr=1000 eps=0.1 att=1000')

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.001)
    _, label, center = cv.kmeans(xy, 2, None, criteria, 1, cv.KMEANS_RANDOM_CENTERS)
    show_data(xy, label, center, axes[1])
    axes[1].set_title('itr=10 eps=0.001 att=1')
    set_label(axes, 'x', 'y')

    figure.suptitle("apply K-means in Cartesian coordinate system")
    figure.set_figheight(5)
    figure.set_figwidth(10)
    figure.savefig('birds2.jpg')

    figure, axes = plt.subplots(1, 2)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)
    _, label, center = cv.kmeans(rt, 2, None, criteria, 1, cv.KMEANS_RANDOM_CENTERS)
    show_data(rt, label, center, axes[0])
    axes[0].set_title('data in Polar coordinate system')
    axes.flat[0].set(xlabel='r', ylabel='\u03B8')

    new_centers = np.empty(center.shape, 'float32')
    new_centers[:, 0] = center[:, 0] * np.cos(center[:, 1])
    new_centers[:, 1] = center[:, 0] * np.sin(center[:, 1])
    show_data(xy, label, center, axes[1])
    axes[1].set_title("back to Cartesian coordinate system")
    axes.flat[1].set(xlabel='x', ylabel='y')

    figure.set_figheight(5)
    figure.set_figwidth(10)
    figure.suptitle("apply K-means in Polar coordinate system")
    figure.savefig('res03.jpg')

