import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RNP')
cv2.setMouseCallback('RNP', RGB)

cap = cv2.VideoCapture('vehicle_count.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0

tracker = Tracker()

vh_down = {}
counter_down = []

vh_up = {}
counter_up = []

# line y coordinates
cy1 = 170
cy2 = 190
offset = 6

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    #    print(px)
    list = []

    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        # list of vehicles needed to be detected
        if 'car' or 'truck' or 'motorcycle' in c:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        if (cy + offset) > cy1 > (cy - offset):
            vh_down[id] = cy
        if id in vh_down:
            if (cy + offset) > cy2 > (cy - offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                if counter_down.count(id) == 0:
                    counter_down.append(id)

        if (cy + offset) > cy2 > (cy - offset):
            vh_up[id] = cy
        if id in vh_up:
            if (cy + offset) > cy1 > (cy - offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                if counter_up.count(id) == 0:
                    counter_up.append(id)

    # Drawing lines
    cv2.line(frame, (110, cy1), (1020, cy1), (255, 255, 255), 1)
    # cv2.putText(frame, '1Line', (224, cy1 - 3), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (70, cy2), (1020, cy2), (255, 255, 255), 1)
    # cv2.putText(frame, '2Line', (116, cy2 - 3), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    down = len(counter_down)
    cv2.putText(frame, ('Going down: ') + str(down), (60, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    up = len(counter_up)
    cv2.putText(frame, ('Going up: ') + str(up), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("RNP", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
