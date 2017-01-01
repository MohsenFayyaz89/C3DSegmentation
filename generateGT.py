import numpy as np
from PIL import Image, ImageDraw


def get_bb(x, y, width, height, angle):
    bb = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_bb = np.dot(bb, R) + offset
    return transformed_bb


def GT(x, y, w, h, sx, sy, label, bg):
    gt = np.ones([sx, sy]) * bg
    gt = Image.fromarray(gt)
    draw = ImageDraw.Draw(gt)
    bb = get_bb(x, y, w, h, 0)
    draw.polygon([tuple(p) for p in bb], fill=label)

    return (gt)
