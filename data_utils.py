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
    """
    landmarks: список з 21 (x,y) координати ключових точок MediaPipe

    Фічі:
    - Одинична база масштабу: відстань від зап’ястя (0) до середнього MCP (9).
    - Нормалізовані відстані від зап’ястя до кінчиків пальців.
    - Нормалізовані відстані між сусідніми кінчиками пальців (4-8, 8-12, 12-16, 16-20).
    - Кут між векторами «зап’ястя→середній суглоб (9)» та «зап’ястя→кінчик пальця» для кожного пальця.
    """
    # 1) Визначаємо базовий масштаб — відстань між зап’ястям і середнім MCP (landmark 9)
    wrist = np.array(landmarks[0])
    mcp_middle = np.array(landmarks[9])
    scale = np.linalg.norm(mcp_middle - wrist) + 1e-6  # +epsilon щоб уникнути ділення на нуль

    # 2) Нормалізовані відстані зап’ястя → кінчики пальців
    tips_idx = [4, 8, 12, 16, 20]
    tip_coords = [np.array(landmarks[i]) for i in tips_idx]
    dists_to_wrist = [np.linalg.norm(tip - wrist) / scale for tip in tip_coords]

    # 3) Відстані між сусідніми кінчиками пальців (4-8, 8-12, 12-16, 16-20), теж нормалізовані
    dists_between_tips = []
    for i in range(len(tip_coords) - 1):
        d = np.linalg.norm(tip_coords[i+1] - tip_coords[i]) / scale
        dists_between_tips.append(d)

    # 4) Кути між вектором wrist→mcp_middle та кожним wrist→tip
    base_vec = mcp_middle - wrist
    base_norm = np.linalg.norm(base_vec)
    angles = []
    for tip in tip_coords:
        vec = tip - wrist
        cos_theta = np.dot(base_vec, vec) / (base_norm * (np.linalg.norm(vec) + 1e-6))
        angles.append(np.arccos(np.clip(cos_theta, -1, 1)))

    # 5) Об’єднуємо всі фічі в один вектор
    features = np.concatenate([
        dists_to_wrist,      # 5
        dists_between_tips,  # 4
        angles               # 5
    ])
    return features

def load_dataset(path):
    features, labels = [], []
    gesture_lst = os.listdir(path)  # ['call', 'fist', 'ok', 'palm', 'rock', 'thumb', 'two', 'numbers']
    
    for gesture_name in gesture_lst:
        if gesture_name.startswith('.'):  # Пропускаємо приховані файли
            continue

        gesture_path = os.path.join(path, gesture_name)
        
        # Обробка вкладеної папки numbers
        if gesture_name == 'numbers':
            numbers_path = os.path.join(path, 'numbers')
            if not os.path.isdir(numbers_path):
                continue
                
            # Отримуємо список цифр (Eight, Five, тощо)
            number_gestures = os.listdir(numbers_path)
            for number_name in number_gestures:
                if number_name.startswith('.'):  # Пропускаємо приховані файли
                    continue
                number_path = os.path.join(numbers_path, number_name)
                process_gesture_folder(number_path, number_name.lower(), features, labels)
        
        # Обробка звичайних жестів
        elif os.path.isdir(gesture_path):
            process_gesture_folder(gesture_path, gesture_name.lower(), features, labels)
    
    return np.array(features), np.array(labels)

def process_gesture_folder(folder_path, gesture_name, features, labels):
    """Допоміжна функція для обробки однієї папки з зображеннями жестів"""
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = get_landmarks(img_rgb)
        if landmarks:
            features.append(extract_features(landmarks))
            labels.append(gesture_name)
        else:
            print(f"Landmarks not found for {img_path}")
            try:
                os.remove(img_path)  # Видаляємо зображення
            except Exception as e:
                print(f"Error deleting {img_path}: {str(e)}")
