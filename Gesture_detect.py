from ultralytics import YOLO
import cv2
import requests as rq

# Load YOLOv8n model
model = YOLO("YOLOv8n_1/YOLOv8ngestureRec.pt")

# initilize cam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera unavailable")
    exit()
while True:
    ret, frame = cam.read()
    #frame error check
    if not ret:
        print("Frame could not be read")
        continue
    #run inference
    results = model.predict(frame,conf=.5,imgsz= 480, max_det=1)
    #process results list
    for result in results:
        boxes = result.boxes.cls.tolist()
        while boxes:
            label = boxes[0]
            print(label)
    #annotate frame
    bound_frame = results[0].plot()
    #display frame
    cv2.imshow("Bounding Boxes", bound_frame)
    #quit condition
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cam.release()