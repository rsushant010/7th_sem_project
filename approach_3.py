# # approach 1

# # define the architecture 
# # move servos only at the desired angles (at the center of the box) using map function
# # after movement trigger fire for 5 sec
# # pause do the prediction in a loop


# # approach 2 (difficult)

# # define the architecture and sketch 
# # if the difference of center of previous detection is more than 20% only then move the servos
# # trigger the fire for better good only after 2 sec if the diff is less than 20% else if 
# # more distance then after detection the fire will trigger better
# # move servos only at the desired angles using map function
# # after detection trigger fire for 5 sec
# # pause do the prediction in a loop




import cv2
from ultralytics import YOLO
import threading
import time
import serial

# Initialize YOLO model
model = YOLO("yolo11s.pt")

# Shared variables between threads
frame_width, frame_height = 640, 480
latest_bbox = None
lock = threading.Lock()  # Lock to synchronize data access


# Initialize serial communication with Arduino
arduinoData = serial.Serial('COM19', 9600, timeout=1)  # Replace 'COM3' with your Arduino port
time.sleep(2)  # Allow some time for the connection to establish

# Variables for idle mode
last_detection_time = time.time()
idle_mode = False
idle_sweep_angle = 0
idle_direction = 1  # 1 for increasing, -1 for decreasing


# Function to send coordinates to Arduino
def send_coordinates_to_arduino(x_center, y_center):
    # Convert the coordinates to a string and send it to Arduino
    coordinates = f"DETECTION {int(x_center)},{int(y_center)}\r"
    arduinoData.write(coordinates.encode())
    print(f"Sent to Arduino: {coordinates}")


def start_survilince():






# def send_idle_to_arduino():
#     global idle_sweep_angle, idle_direction

#     # Increment or decrement the sweep angle
#     idle_sweep_angle += idle_direction * 5  # Adjust the step size if necessary

#     # Reverse direction if limits are reached
#     if idle_sweep_angle >= 180 or idle_sweep_angle <= 0:
#         idle_direction *= -1

#     # Send idle command to Arduino
#     idle_command = f"IDLE {idle_sweep_angle}\r"
#     arduinoData.write(idle_command.encode())
#     print(f"Idle Mode: {idle_command}")
    



# Initialize previous detection center coordinates
prev_x_center, prev_y_center = None, None


# Object detection function
def detect_objects(cam_num):
    
    global latest_bbox, prev_x_center, prev_y_center
    cap = cv2.VideoCapture(cam_num)  # Use camera

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No video frame available")
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        results = model.predict(source=frame, conf=0.75, save=False)



        for box in results[0].boxes:
            clsID = int(box.cls[0])  # Class ID
            conf = float(box.conf[0])
            class_name = model.names[clsID]

            # Check if detected object is 'bird' or 'person'
            if 'bird' in class_name.lower() or 'person' in class_name.lower():
                bb = box.xyxy[0]  # Bounding box
                x_center = (bb[0] + bb[2]) / 2
                y_center = (bb[1] + bb[3]) / 2

                y = int((y_center - 0) * (0 - 180) / (480 - 0) + 180)
                x = int((x_center - 0) * (0 - 180) / (640 - 0) + 180)

                send_coordinates_to_arduino(y, x)

                last_detection_time = time.time()  # Update detection time
                detected = True
                
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f" {class_name , round(conf * 100, 2)}%",           # Text to display
                    (int(bb[0]), int(bb[1]) - 10),          # Position (slightly above the bounding box)
                        cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,                                     # Font scale
                    (255, 255, 255),                         # Font color
                    1,                                       # Font thickness
                    )

                
                # with lock:
                #     latest_bbox = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))


            
            # Handle idle mode if no detection for 5 seconds
            if class_name.lower() == 'no detections':
                start_survilince(90,90)


               
        cv2.imshow("Object Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the detection thread
detection_thread = threading.Thread(target=detect_objects, args=(0,))
detection_thread.start()
detection_thread.join()

