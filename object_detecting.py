import cv2
import supervision as sv
from ultralytics import YOLO

path = "E:\\Fruit Object Detection Project\\Object Detection Project\\runs6\\content\\runs\\detect\\train2\\weights\\best.pt"
model = YOLO(path)

cap = cv2.VideoCapture(0)

boundingBoxAnnotator = sv.BoundingBoxAnnotator()
labelAnnotator = sv.LabelAnnotator()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        annotatedFrame = boundingBoxAnnotator.annotate(scene=frame, detections=detections)
        annotatedFrame = labelAnnotator.annotate(scene=annotatedFrame, detections=detections)

        cv2.imshow("Live Object Detection - YOLO11", annotatedFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()