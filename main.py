import cv2
from ultralytics import YOLO
import numpy as np
import torch 

cap = cv2.VideoCapture(0)


model = YOLO("yolomodel/ust_model.pt")
tolerance=0.1


while True:
    ret , frame = cap.read()
    if not ret:
        break
    #Change according to processors(cpu,gpu,macbook)
    height, width = frame.shape[:2]
    frame_center_x = width // 2
    frame_center_y = height // 2

    # Calculate the tolerance zone coordinates
    tolerance_x1 = int(width / 2 - tolerance * width)
    tolerance_y1 = int(height / 2 - tolerance * height)
    tolerance_x2 = int(width / 2 + tolerance * width)
    tolerance_y2 = int(height / 2 + tolerance * height)
    
    # Draw the tolerance zone
    cv2.rectangle(frame, (int(width/2-tolerance*width),int(height/2-tolerance*height)), (int(width/2+tolerance*width),int(height/2+tolerance*height)), (0,255,0), 2)

    results = model(frame, device="mps") 
    result = results[0]

    bboxes = np.array(result.boxes.xyxy.cpu() , dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    
    #draw center lines
    cv2.line(frame, (frame_center_x, 0), (frame_center_x, height), (0, 255, 0), 2)
    cv2.line(frame, (0, frame_center_y), (width, frame_center_y), (0, 255, 0), 2)

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        object_center_x = (x + x2) // 2
        object_center_y = (y + y2) // 2
        
        # Normalize the center coordinates
        norm_x = (object_center_x - frame_center_x) / (frame_center_x if frame_center_x != 0 else 1)
        norm_y = (object_center_y - frame_center_y) / (frame_center_y if frame_center_y != 0 else 1)
       
        # Draw bounding box and class label
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        # Display the normalized center coordinates
        cv2.putText(frame, f"({norm_x:.2f}, {norm_y:.2f})", (object_center_x, object_center_y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Determine the movement direction based on normalized deviations
        if abs(norm_x) < tolerance and abs(norm_y) < tolerance:
            direction = "Stop"
        elif abs(norm_x) > abs(norm_y):
            direction = "Move Left" if norm_x >= tolerance else "Move Right"
        else:
            direction = "Move Forward" if norm_y >= tolerance else "Move Backward"

        # Display the direction on the frame
        cv2.putText(frame, direction, (object_center_x, object_center_y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
        
        # if tolerance_x1 <= object_center_x <= tolerance_x2 and tolerance_y1 <= object_center_y <= tolerance_y2:
        #     cv2.putText(frame, "In Zone", (object_center_x, object_center_y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        # else:
        #     cv2.putText(frame, "Out of Zone", (object_center_x, object_center_y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("Detection", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()