import cv2
import joblib
import mediapipe as mp
from collections import deque, Counter
from landmarks import normalize_landmarks
clf = joblib.load("models/gesture_rf.pkl")
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,min_tracking_confidence=0.7)
history = deque(maxlen=8)
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    label = "no hand"
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks,
        mp_hands.HAND_CONNECTIONS)
        features = normalize_landmarks(hand_landmarks).reshape(1, -1)
        prediction = clf.predict(features)[0]
        history.append(prediction)
        label = Counter(history).most_common(1)[0][0]
    cv2.putText(frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 200, 0), 3)
    cv2.imshow("Gesture Recognizer - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()