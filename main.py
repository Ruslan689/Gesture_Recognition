import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from data_utils import load_dataset, extract_features, mp_hands
from svm import ManualSVM

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from collections import Counter


def data_model():
    DATA_ROOT = "/Users/ruslan/Desktop/Лінійна_алгебра/project/try2/data"

    # 1) Завантажуємо всі дані
    X, y = load_dataset(DATA_ROOT)

    # 2) Кодуємо та масштабуємо
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # 3) Train/Test спліт
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # 4) Навчаємо SVM
    svm = ManualSVM(C=1.0, gamma=0.1)
    svm.fit(X_train, y_train_enc)

    # 5) Прогноз і метрики
    y_pred_enc = svm.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    y_test = le.inverse_transform(y_test_enc)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # 6) handmade пер-класово
    print("\n=== Handmade confusion matrix ===\n")
    for idx in range(len(le.classes_)):
        handmade_confusion_matrix(svm, X_test, y_test_enc, le, idx)

    return le, scaler, svm

def real_time_detection(le, scaler, svm):
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=5, min_detection_confidence=0.7)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in results.multi_hand_landmarks:
                # Отримуємо координати
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                features = scaler.transform([extract_features(landmarks)])
                pred = svm.predict(features)
                label = le.inverse_transform(pred)[0]

                # Малюємо landmark'и
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Обчислюємо межі руки
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Малюємо прямокутник
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 215, 255), 2)

                # Пишемо назву жесту
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 0, 0), 3)

        cv2.imshow('Hand Gesture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def handmade_confusion_matrix(model, X_test, y_test, le, class_idx):
    class_name = le.classes_[class_idx]
    
    # Перетворюємо всі предикти на бінарні: цей клас — 1, всі інші — 0
    y_true_binary = (y_test == class_idx).astype(int)
    y_pred_all = model.predict(X_test)
    y_pred_binary = (y_pred_all == class_idx).astype(int)

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    # Гарантуємо, що отримаємо 2x2 матрицю
    if cm.shape != (2, 2):
        # Якщо всі предикти або всі справжні мітки = 0 (немає позитивних прикладів)
        tn = cm[0][0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0][1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1][0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = cm[1][1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    else:
        tn, fp, fn, tp = cm.ravel()

    # Метрики
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    sensitivity = tp / (tp + fn + 1e-10)  # Recall
    specificity = tn / (tn + fp + 1e-10)

    # Вивід
    print(f"\n=== Confusion matrix for class «{class_name}» ===")
    print(f"{'':15} predicted_0  predicted_1")
    print(f"actual_0{tn:13}{fp:13}")
    print(f"actual_1{fn:13}{tp:13}")
    print(f"Accuracy   = {accuracy:.3f}")
    print(f"Sensitivity= {sensitivity:.3f}")
    print(f"Specificity= {specificity:.3f}")


if __name__ == '__main__':
    le, scaler, svm = data_model()
    real_time_detection(le, scaler, svm)
