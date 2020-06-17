import numpy as np
from scipy.optimize import linear_sum_assignment


def boxes_iou(boxes_a, boxes_b) -> np.ndarray:
    """计算两组boxes两两之间的IOU值, box的格式必须是：x1, y1, x2, y2

    Parameters
    ----------
    boxes_a : np.ndarray, M x 4
        第1组boxes
    boxes_b : np.ndarray, N x 4
        第2组boxes

    Returns
    -------
    np.ndarray
        M x N 二维数组
    """
    if boxes_a.shape[1] != 4 or boxes_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    # bottom right
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(boxes_a[:, 2:] - boxes_a[:, :2], axis=1)
    area_b = np.prod(boxes_b[:, 2:] - boxes_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


class IOUTrack(object):
    def __init__(self, iou_threshold=0.7, miss_threshold=3):
        """
        Parameters
        ----------
        iou_threshold : float, optional
            矩形框重叠阈值，新输入的矩形框和已有轨迹最后一个位置的矩形框，IOU大于iou_threshold
            则认为两者可以匹配上，否则两者不能进行匹配。
            by default 0.7
        miss_threshold : int, optional
            如果一个轨迹连续miss_threshold帧(大于等于)不能更新，则该轨迹失效
            by default 0.7
        """
        self._iou_threshold = iou_threshold
        self._miss_threshold = miss_threshold
        self._trajectories = list()
        self._frame_index = 0

    def _assignment(self, prev_boxes, curr_boxes):
        matches = list()
        unmatched_tracks, unmatched_detects = list(), list()
        if len(prev_boxes) == 0:
            unmatched_detects = list(range(len(curr_boxes)))
        elif len(curr_boxes) == 0:
            unmatched_tracks = list(range(len(prev_boxes)))
        else:
            iou = boxes_iou(prev_boxes, curr_boxes)
            row_ind, col_ind = linear_sum_assignment(-iou)
            for i in range(prev_boxes.shape[0]):
                if i not in row_ind:
                    unmatched_tracks.append(i)
            for i in range(curr_boxes.shape[0]):
                if i not in col_ind:
                    unmatched_detects.append(i)
            for i, j in zip(row_ind, col_ind):
                if iou[i, j] >= self._iou_threshold:
                    matches.append([i, j])
                else:
                    unmatched_tracks.append(i)
                    unmatched_detects.append(j)
        matches = np.array(matches)
        unmatched_tracks = np.array(unmatched_tracks)
        unmatched_detects = np.array(unmatched_detects)
        return matches, unmatched_tracks, unmatched_detects

    def _update_trajectories(self, image, matches, unmatched_tracks,
                             unmatched_detects, prev_box_index_to_tra_id,
                             curr_boxes):
        # 已有轨迹在当前帧能够找到匹配上的矩形框
        for match in matches:
            matched_track_index, matched_detect_index = match
            matched_tra_id = prev_box_index_to_tra_id[matched_track_index]
            for trajectory in self._trajectories:
                if trajectory["id"] == matched_tra_id:
                    trajectory["boxes"].append(
                        curr_boxes[matched_detect_index])
                    trajectory["images"].append(image)
                    trajectory["frame_index"].append(self._frame_index)
        # 已有轨迹在当前帧不能找到匹配上的矩形框
        for unmatched_track in unmatched_tracks:
            unmatched_tra_id = prev_box_index_to_tra_id[unmatched_track]
            for trajectory in self._trajectories:
                if trajectory["id"] == unmatched_tra_id:
                    trajectory["miss_count"] += 1
        # 新出现的矩形框
        for unmatched_detect in unmatched_detects:
            new_trajectory = dict()
            new_trajectory["id"] = 1 if len(self._trajectories) == 0 \
                else self._trajectories[-1]["id"] + 1
            new_trajectory["boxes"] = [curr_boxes[unmatched_detect]]
            new_trajectory["images"] = [image]
            new_trajectory["frame_index"] = [self._frame_index]
            new_trajectory["miss_count"] = 0
            self._trajectories.append(new_trajectory)

    def track(self, boxes, scores, image):
        prev_boxes = list()
        prev_box_index_to_tra_id = dict()
        for trajectory in self._trajectories:
            tra_id = trajectory["id"]
            prev_box = trajectory["boxes"][-1]
            miss_count = trajectory["miss_count"]
            if miss_count > self._miss_threshold:
                continue
            prev_boxes.append(prev_box)
            prev_box_index_to_tra_id[len(prev_boxes) - 1] = tra_id
        prev_boxes = np.array(prev_boxes)
        matches, unmatched_tracks, unmatched_detects = \
            self._assignment(prev_boxes, boxes)
        self._update_trajectories(image, matches, unmatched_tracks, unmatched_detects,
                                  prev_box_index_to_tra_id, boxes)
        self._frame_index += 1
        boxes, tra_ids = list(), list()
        for trajectory in self._trajectories:
            if trajectory["miss_count"] <= self._miss_threshold:
                boxes.append(trajectory["boxes"][-1])
                tra_ids.append(trajectory["id"])
        return boxes, tra_ids

    @property
    def trajectories(self):
        """
        {
            'id':1, 
            'boxes':[<box1>,<box2>,...],
            'images':[<image1>, <image2>], 
            'frame_index':[0,1,...],
            'miss_count':0
            }
        """
        return self._trajectories

    def release(self):
        self._trajectories = list()
