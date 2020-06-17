import os
import sys
root_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(root_dir)
import numpy as np
import cv2
from kcf_tracker import KCFTracker, draw_detection_rects


def tracker(cam, frame, bbox):
    tracker = KCFTracker(True, True, True) # (hog, fixed_Window, multi_scale)
    tracker.init(bbox, frame)
    
    while True:
        ok, frame = cam.read()

        timer = cv2.getTickCount()
        bbox = tracker.update(frame)
        bbox = list(map(int, bbox))
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        # Put FPS
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tracker = KCFTracker()
    image_path = os.path.dirname(os.path.dirname(root_dir)) + "/data/image_data/images/00000_1.jpg"
    print(image_path)
    image = cv2.imread(image_path)
    # left, top, width, height
    boxes = [146, 193, 371, 371]
    tracker.init(boxes, image)
    for index in range(5):
        frame = image.copy()
        bbox = tracker.update(frame) 
        print(bbox)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        cv2.putText(frame, str(index), (50,50),1,2,(0,255,0))
        cv2.imshow("track", frame)
        cv2.waitKey(0)

