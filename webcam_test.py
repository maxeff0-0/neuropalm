# Program to test the working of webcam. Can be run before the actual recognition file to verify the working of hardware
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open the webcam.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    cv2.imshow("Webcam test. Press q to quit.", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
