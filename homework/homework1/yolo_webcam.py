import time
import cv2
import numpy as np

WEIGHTS = "yolo3/yolov3.weights"
CONFIG = "yolo3/yolov3.cfg"
NAMES = "yolo3/coco.names"
INPUT_SIZE = 416
CONFIDENCE = 0.5
NMS_THRESHOLD = 0.4

net = cv2.dnn.readNet(WEIGHTS, CONFIG)
with open(NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN


def detect(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (INPUT_SIZE, INPUT_SIZE), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                boxes.append([int(center_x - w / 2), int(center_y - h / 2), w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, NMS_THRESHOLD)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "%s-%3.1f" % (classes[class_ids[i]], confidences[i])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 2, color, 2)
    return frame


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
start = time.time()
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    frame_count += 1
    frame = detect(cv2.flip(frame, 1))
    fps = frame_count / (time.time() - start)
    cv2.putText(frame, "fps %.2f" % fps, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Yolo v3  (q: quit)", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
        break
cap.release()
cv2.destroyAllWindows()
