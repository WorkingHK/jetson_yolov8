import cv2
from ultralytics import YOLO
import numpy as np

# Constants
TOLERANCE = 0.1
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_COLOUR = (0, 255, 0)

# Function to initialize the YOLO model
def initialize_model(model_path):
    return YOLO(model_path)

# Function to capture a frame from the video feed
def get_frame(cap):
    ret, frame = cap.read()
    return ret, frame

# Function to draw the tolerance zone on the frame
def draw_tolerance_zone(frame, width, height):
    tolerance_x1 = int(width / 2 - TOLERANCE * width)
    tolerance_y1 = int(height / 2 - TOLERANCE * height)
    tolerance_x2 = int(width / 2 + TOLERANCE * width)
    tolerance_y2 = int(height / 2 + TOLERANCE * height)
    cv2.rectangle(frame, (tolerance_x1, tolerance_y1), (tolerance_x2, tolerance_y2), (0, 255, 0), 2)

# Function to draw the center lines on the frame
def draw_center_lines(frame, width, height):
    frame_center_x = width // 2
    frame_center_y = height // 2
    cv2.line(frame, (frame_center_x, 0), (frame_center_x, height), TEXT_COLOUR, 2)
    cv2.line(frame, (0, frame_center_y), (width, frame_center_y), TEXT_COLOUR, 2)

# Function to normalize the coordinates of the object's center
def normalize_coordinates(object_center, frame_center):
    return (object_center - frame_center) / (frame_center if frame_center != 0 else 1)

# Function to determine the movement direction based on normalized deviations
def get_direction(norm_x, norm_y, tolerance):
    if abs(norm_x) < tolerance and abs(norm_y) < tolerance:
        return "Stop and grab"
    if abs(norm_x) > abs(norm_y):
        return "Move Left" if norm_x >= tolerance else "Move Right"
    return "Move Forward" if norm_y >= tolerance else "Move Backward"

# Function to process the frame for object detection and determine movement direction
def process_frame(frame, model, width, height):
    frame_center_x = width // 2
    frame_center_y = height // 2
    
    # Perform object detection using YOLO model
    results = model(frame, device="mps")
    result = results[0]
    
    # Get bounding boxes and classes of detected objects
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    
    # Draw tolerance zone and center lines
    draw_tolerance_zone(frame, width, height)
    draw_center_lines(frame, width, height)
    
    # Process each detected object
    for cls, bbox in zip(classes, bboxes):
        x, y, x2, y2 = bbox
        object_center_x = (x + x2) // 2
        object_center_y = (y + y2) // 2
        
        # Normalize the object's center coordinates
        norm_x = normalize_coordinates(object_center_x, frame_center_x)
        norm_y = normalize_coordinates(object_center_y, frame_center_y)
        
        # Draw bounding box and class label on the frame
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, str(cls), (x, y - 5), FONT, 2, (0, 0, 255), 2)
        cv2.putText(frame, f"({norm_x:.2f}, {norm_y:.2f})", (object_center_x, object_center_y - 10), FONT, 2, TEXT_COLOUR, 2)
        
        # Determine and display the movement direction
        direction = get_direction(norm_x, norm_y, TOLERANCE)
        cv2.putText(frame, direction, (object_center_x, object_center_y + 20), FONT, 2, TEXT_COLOUR, 2)
        cv2.putText(frame, direction, (frame_center_x + 10, height - 8), FONT, 2, TEXT_COLOUR, 2)

# Main function to execute the video capture and processing loop
def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Initialize YOLO model
    model = initialize_model("yolomodel/ust_model.pt")
    
    while True:
        # Capture a frame from the video feed
        ret, frame = get_frame(cap)
        if not ret:
            break
        
        # Get the frame dimensions
        height, width = frame.shape[:2]
        
        # Process the frame for object detection and movement direction
        process_frame(frame, model, width, height)
        
        # Display the processed frame
        cv2.imshow("Detection", frame)
        
        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(1) == 27:
            break
    
    # Release video capture and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Execute the main function
if __name__ == "__main__":
    main()
