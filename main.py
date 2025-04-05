import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler, LabelEncoder

from try2.Gesture_Recognition.data_utils import load_dataset, extract_features, mp_hands
from try2.Gesture_Recognition.svm import ManualSVM


def data_model():
    X, y = load_dataset("root_dataset")
    le = LabelEncoder().fit(y)
    y_encoded = le.transform(y)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    svm = ManualSVM(C=1.0, gamma=0.1)
    svm.fit(X_scaled, y_encoded)
    print("Model is trained.")
    return le, scaler, svm

def real_time_detection(le, scaler, svm):
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            landmarks = [(lm.x, lm.y) for lm in results.multi_hand_landmarks[0].landmark]
            features = scaler.transform([extract_features(landmarks)])
            pred = svm.predict(features)
            label = le.inverse_transform(pred)[0]

            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    le, scaler, svm = data_model()
    real_time_detection(le, scaler, svm)
