import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

# Load YOLO model
model = YOLO('yolov8s.pt')


# Mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


# Set up mouse callback
cv2.namedWindow('RNP')
cv2.setMouseCallback('RNP', RGB)

# Open video capture
cap = cv2.VideoCapture('new_car_video.mp4')

# Read class names from a file
with open("coco.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

# Initialize tracker
tracker = Tracker()

# Define dictionaries and counters for vehicle tracking
vh_down = {}
counter_down = []
vh_up = {}
counter_up = []

# Line y coordinates
cy1 = 170
cy2 = 190
offset = 6

# Initialize count variable
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for faster processing
    count += 1
    if count % 3 != 0:
        continue

    # Resize the frame
    frame = cv2.resize(frame, (1020, 500))

    # Predict with YOLO model
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Extract relevant information about vehicles
    vehicles = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]
        if 'car' in c or 'truck' in c or 'motorcycle' in c:
            vehicles.append([x1, y1, x2, y2])

    # Update tracker and process tracked vehicles
    bbox_id = tracker.update(vehicles)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        # Check if the vehicle crosses the lines
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

    # Draw lines
    cv2.line(frame, (110, cy1), (1020, cy1), (255, 255, 255), 1)
    cv2.line(frame, (70, cy2), (1020, cy2), (255, 255, 255), 1)

    # Display counts
    down = len(counter_down)
    cv2.putText(frame, ('Going down: ') + str(down), (60, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    up = len(counter_up)
    cv2.putText(frame, ('Going up: ') + str(up), (794, 77), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("RNP", frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
