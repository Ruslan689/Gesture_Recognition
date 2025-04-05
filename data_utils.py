import os
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands_static = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def get_landmarks(image):
    results = hands_static.process(image)
    if results.multi_hand_landmarks:
        return [(lm.x, lm.y) for lm in results.multi_hand_landmarks[0].landmark]
    return None

def extract_features(landmarks):
    # Distance from wrist to fingertips
    wrist = landmarks[0]
    tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]
    distances = [np.hypot(tip[0] - wrist[0], tip[1] - wrist[1]) for tip in tips]

    # Angles between finger vectors
    vectors = [(tip[0] - wrist[0], tip[1] - wrist[1]) for tip in tips]
    angles = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            cos_theta = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-10)
            angles.append(np.arccos(np.clip(cos_theta, -1, 1)))
    return np.concatenate([distances, angles])

def load_dataset(path):
    features, labels = [], []
    gesture_lst = ['palm', 'l', 'fist', 'thumb', 'index', 'ok', 'c']

    for subject in os.listdir(path):
        subject_path = os.path.join(path, subject)
        if not os.path.isdir(subject_path): 
            continue

        for gesture_folder in os.listdir(subject_path):
            gesture_path = os.path.join(subject_path, gesture_folder)
            if '_' not in gesture_folder:
                continue

            gesture_name = gesture_folder.split('_')[1].lower()
            if gesture_name not in gesture_lst:
                continue

            for img_file in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                landmarks = get_landmarks(img_rgb)
                if landmarks:
                    features.append(extract_features(landmarks))
                    labels.append(gesture_name)

    return np.array(features), np.array(labels)
