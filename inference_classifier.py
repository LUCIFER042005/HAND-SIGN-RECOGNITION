import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# --- AUTOMATIC VARIABLES ---
sentence = ""
auto_mode = False
last_add_time = time.time()
COLLECTION_INTERVAL = 3.0

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = ""

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = str(prediction[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            # --- AUTOMATIC TIMER LOGIC ---
            if auto_mode:
                current_time = time.time()
                elapsed_time = current_time - last_add_time

                remaining = max(0, COLLECTION_INTERVAL - elapsed_time)
                cv2.putText(frame, f"Next in: {remaining:.1f}s", (W - 250, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if elapsed_time >= COLLECTION_INTERVAL:
                    if predicted_character:

                        # --- SPACE AND DELETE LOGIC ---
                        label_upper = predicted_character.upper()

                        if label_upper == "SPACE":
                            sentence += " "
                        elif label_upper == "DELETE":
                            sentence = sentence[:-1]
                        else:
                            sentence += predicted_character

                        # --- UPDATED TERMINAL OUTPUT ---
                        print("-" * 30)
                        print(f"Current Sentence: {sentence}")
                        print("-" * 30)

                        last_add_time = current_time  # Reset timer


        except Exception as e:
            pass

    # UI: Draw Sentence Bar
    cv2.rectangle(frame, (0, H - 80), (W, H), (255, 255, 255), -1)
    cv2.putText(frame, f"Sentence: |{sentence}|", (20, H - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)

    # UI: Status Bar
    status_color = (0, 255, 0) if auto_mode else (0, 0, 255)
    status_text = "AUTO ON" if auto_mode else "AUTO OFF (Press S)"
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow('frame', frame)

    key_press = cv2.waitKey(1) & 0xFF
    if key_press == ord('q'):
        break
    elif key_press == ord('s'):
        auto_mode = not auto_mode
        last_add_time = time.time()
    elif key_press == ord('c'):
        sentence = ""
        print("\nSentence Cleared!\n")

cap.release()
cv2.destroyAllWindows()