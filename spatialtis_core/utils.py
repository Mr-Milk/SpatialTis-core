from typing import List


def show_options(input: str, options: List):
    return f"{input} not found, available options are {', '.join([str(i) for i in options])}"


def default_radius(bbox, ratio=0.1):
    w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
    r = min([w, h]) * ratio
    return r


def default_radius_3d(bbox, ratio=0.1):
    w, h, d = (bbox[3] - bbox[0]), (bbox[4] - bbox[1]), (bbox[5] - bbox[2])
    r = min([w, h, d]) * ratio
    return r
