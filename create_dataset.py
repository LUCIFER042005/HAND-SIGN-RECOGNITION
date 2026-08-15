import os
import pickle
import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []


def normalize_landmarks(landmarks):
    """Centers coordinates around the wrist and scales by maximum span."""
    # Landmark 0 is the wrist
    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y

    relative_coords = []
    for lm in landmarks:
        relative_coords.append([lm.x - wrist_x, lm.y - wrist_y])

    relative_coords = np.array(relative_coords)

    # Calculate max Euclidean distance from the wrist to normalize scale
    max_dist = np.max(np.sqrt(np.sum(relative_coords ** 2, axis=1)))
    if max_dist == 0:
        max_dist = 1e-6

    normalized = relative_coords / max_dist
    return normalized


def augment_landmarks(normalized_coords):
    """Generates synthetic variations: rotation, jitter, scale, and mirroring."""
    augmented_samples = []

    # 1. Original sample
    augmented_samples.append(normalized_coords.flatten().tolist())

    # 2. Left/Right Hand Mirrored version (Invert X-axis)
    mirrored = normalized_coords.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    augmented_samples.append(mirrored.flatten().tolist())

    # 3. Small rotations (+-8 degrees, +-15 degrees)
    angles = [-15, -8, 8, 15]
    for angle in angles:
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Rotate original
        rotated = np.dot(normalized_coords, rot_matrix)
        augmented_samples.append(rotated.flatten().tolist())

        # Rotate mirrored
        rotated_mirrored = np.dot(mirrored, rot_matrix)
        augmented_samples.append(rotated_mirrored.flatten().tolist())

    # 4. Slight Gaussian noise (simulates webcam shake / varying hand landmarks)
    for _ in range(2):
        noise = np.random.normal(0, 0.015, normalized_coords.shape)
        noisy_sample = normalized_coords + noise
        augmented_samples.append(noisy_sample.flatten().tolist())

    return augmented_samples


print("Processing dataset with scale normalization & data augmentation...")

for dir_ in sorted(os.listdir(DATA_DIR)):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    print(f"Extracting features for class: {dir_}")
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Normalize hand points
                normalized_points = normalize_landmarks(hand_landmarks.landmark)

                # Apply data augmentation
                sample_variations = augment_landmarks(normalized_points)

                for sample in sample_variations:
                    data.append(sample)
                    labels.append(dir_)

print(f"\nExtraction complete! Total training samples: {len(data)}")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Saved processed features to 'data.pickle'.")