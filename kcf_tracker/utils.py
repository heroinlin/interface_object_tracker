# -*- coding: utf-8 -*-
import cv2
import numpy as np


def change_box_order(boxes, order):
    """
    change box order between (xmin, ymin, xmax, ymax) and (center_x, center_y, width, height)

    Args:
        boxes: (numpy array), bounding boxes, size [N, 4]
        order: (str), 'xyxy2xywh' or 'xywh2xyxy'
    Returns:
        converted bounding boxes, size [N, 4]
    """
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    if order == 'xyxy2xywh':
        boxes[:, :2] = (boxes[:, :2] + boxes[:, 2:4]) / 2
        boxes[:, 2:4] -= boxes[:, :2]
    else:
        boxes[:, :2] -= boxes[:, 2:4] / 2
        boxes[:, 2:4] += boxes[:, :2]
    return boxes


def box_transform(bounding_boxes, width, height):
    """
    bounding_boxes  [[score, box],[score, box]]
    box框结果值域由[0,1],[0,1] 转化为[0,width]和[0,height]
    """
    for i in range(len(bounding_boxes)):
        x1 = float(bounding_boxes[i][1])
        y1 = float(bounding_boxes[i][2])
        x2 = float(bounding_boxes[i][3])
        y2 = float(bounding_boxes[i][4])
        bounding_boxes[i][1] = x1 * width
        bounding_boxes[i][2] = y1 * height
        bounding_boxes[i][3] = x2 * width
        bounding_boxes[i][4] = y2 * height
    return bounding_boxes


def draw_detection_rects(image: np.ndarray, detection_rects: np.ndarray, color=(0, 255, 0), method=1):
    if not isinstance(detection_rects, np.ndarray):
        detection_rects = np.array(detection_rects)
    if method:
        width = image.shape[1]
        height = image.shape[0]
    else:
        width = 1.0
        height = 1.0
    for index in range(detection_rects.shape[0]):
        cv2.rectangle(image,
                      (int(detection_rects[index, 0] * width), int(detection_rects[index, 1] * height)),
                      (int(detection_rects[index, 2] * width), int(detection_rects[index, 3] * height)),
                      color,
                      thickness=2)
        if detection_rects.shape[1] == 5:
            cv2.putText(image, f"{detection_rects[index, 4]:.03f}",
                        (int(detection_rects[index, 0] * width), int(detection_rects[index, 1] * height)),
                        1, 1, (255, 0, 255))
