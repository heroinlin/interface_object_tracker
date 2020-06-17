import os
import sys
root_dir = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(root_dir)
import numpy as np
import cv2
from iou_tracker import IOUTrack, draw_detection_rects


if __name__ == '__main__':
    tracker = IOUTrack()
    image_path = os.path.dirname(os.path.dirname(root_dir)) + "/data/image_data/images/00000_1.jpg"
    print(image_path)
    image = cv2.imread(image_path)
    # x1,y1,x2,y2
    boxes = np.array([[146, 193, 517, 564]])
    method = 0
    # boxes = np.array([[0.219, 0.193, 0.776, 0.565]])
    # method = 1
    # scores = np.array([0.9])
    scores = None
    for _ in range(5):
        tracker.track(boxes, scores, image) 
    trajectories = []
    for trajectory in tracker.trajectories:
        trajectories.append(trajectory)
        for index, box in enumerate(trajectory['boxes']):
            frame = trajectory['images'][index].copy()
            draw_detection_rects(frame, [box], method=method)
            cv2.putText(frame, str(trajectory['frame_index'][index]), (50,50),1,2,(0,255,0))
            cv2.imshow("track", frame)
            cv2.waitKey(0)

