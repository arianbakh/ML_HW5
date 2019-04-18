import os
import cv2
import numpy as np


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, 'output.png')


# Algorithm Parameters
K = 3
NUMBER_OF_POINTS = 1000
MAX_EPOCHS = 100


# Image Settings
IMAGE_SHAPE = (512, 512, 3)
BACKGROUND_COLOR = (255, 255, 255)
CLUSTER_POINT_WIDTH = 2
CLUSTER_COLORS = [
    (255, 127, 127),
    (127, 255, 127),
    (127, 127, 255),
]
CENTER_POINT_WIDTH = 4
CENTER_POINT_COLOR = (0, 0, 0)


def _get_random_points():
    return np.random.rand(NUMBER_OF_POINTS, 2)


def _init_center_points(points):
    # Forgy method
    return points[np.random.choice(points.shape[0], K, replace=False), :]


def _get_empty_r():
    return np.zeros((NUMBER_OF_POINTS, K))


def _calc_r(points, center_points):
    r = np.zeros((NUMBER_OF_POINTS, K))
    for n, point in enumerate(points):
        min_distance = 2
        min_k = -1
        for j, center_point in enumerate(center_points):
            distance = np.linalg.norm(point - center_point)
            if distance < min_distance:
                min_distance = distance
                min_k = j
        r[n, min_k] = 1
    return r


def _calc_center_points(points, r):
    center_points = np.zeros((K, 2))
    for k in range(K):
        for n in range(NUMBER_OF_POINTS):
            center_points[k] += r[n, k] * points[n]
        center_points[k] /= sum(r[:, k])  # may have division by zero in edge cases
    return center_points


def _save_output_image(points, center_points, r):
    image = np.full(IMAGE_SHAPE, BACKGROUND_COLOR, dtype=np.uint8)
    for n, point in enumerate(points):
        for k in range(K):
            if r[n, k]:
                cv2.circle(image,
                           (
                               int(point[0] * IMAGE_SHAPE[0]),
                               int(point[1] * IMAGE_SHAPE[1])
                           ),
                           CLUSTER_POINT_WIDTH,
                           CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
                           thickness=-1)
                break  # for performance
    for center_point in center_points:
        cv2.circle(image,
                   (
                       int(center_point[0] * IMAGE_SHAPE[0]),
                       int(center_point[1] * IMAGE_SHAPE[1])
                   ),
                   CENTER_POINT_WIDTH,
                   CENTER_POINT_COLOR,
                   thickness=-1)
    cv2.imwrite(OUTPUT_PATH, image)


def run():
    points = _get_random_points()
    center_points = _init_center_points(points)
    r = _get_empty_r()
    for i in range(MAX_EPOCHS):
        previous_r = r
        r = _calc_r(points, center_points)
        if np.allclose(previous_r, r):
            print('Converged in %d epochs.' % i)
            break
        center_points = _calc_center_points(points, r)
    _save_output_image(points, center_points, r)


if __name__ == '__main__':
    run()
