#I ran this code locally because colab doesn't recognize the cv2.VideoCapture(0) so i ran locally to gather pictures
import cv2
import os

# Open the default camera
cap = cv2.VideoCapture(0)

# Set the frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set the folder to save images
folder_path = "dataCamera/channelUp"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Initialize the counter
img_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for key press
    key = cv2.waitKey(1)

    # Take a snapshot if 's' is pressed
    if key == ord('s'):
        img_name = f"{folder_path}/imgChUp_{img_count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_count += 1

    # Exit if 'q' is pressed
    if key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
