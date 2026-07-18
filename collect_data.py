# Program to collect data regarding the gestures specified
import csv
import os
import cv2
import mediapipe as mp
from landmarks import normalize_landmarks
GESTURES = ["fist", "open_palm", "peace", "thumbs_up", "ok_sign", "point"]
current_label = 0
os.makedirs("data", exist_ok=True)
file_exists = os.path.exists("data/gesture_data.csv")
f = open("data/gesture_data.csv", "a", newline="")
writer = csv.writer(f)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)
saved_count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    hand_landmarks = None
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks,
        mp_hands.HAND_CONNECTIONS)
    info = (f"[{GESTURES[current_label]}] saved={saved_count} "f"0-{len(GESTURES)-1}=select s=save q=quit")
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.55, (0, 255, 0), 2)
    cv2.imshow("Collect Gesture Data", frame)
    key = cv2.waitKey(1) & 0xFF
    if ord("0") <= key <= ord(str(len(GESTURES) - 1)):
        current_label = key - ord("0")
    elif key == ord("s") and hand_landmarks is not None:
        row = normalize_landmarks(hand_landmarks).tolist()
        row.append(GESTURES[current_label])
        writer.writerow(row)
        saved_count += 1
    elif key == ord("q"):
        break
f.close()
cap.release()
cv2.destroyAllWindows()
