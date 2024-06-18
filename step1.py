import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(max_num_hands=2, static_image_mode=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)


DATA_DIR = 'C:\\Users\\KIIT\\Desktop\\College\\6thSem\\MinorProj\\MinorProject\\datasets'

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
        # Construct full image path
        full_img_path = os.path.join(DATA_DIR, dir_, img_path)

        # Check if file exists before attempting to read
        if os.path.exists(full_img_path):
            img = cv2.imread(full_img_path)
            if img is not None:  # Handle cases where image reading fails
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            img_rgb,  # image to draw
                            hand_landmarks,  # model output
                            mp_hands.HAND_CONNECTIONS,  # hand connections
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                plt.figure()
                plt.imshow(img_rgb)
            else:
                print(f"Error: Failed to read image {full_img_path}")  # Inform user about error
        else:
            print(f"Error: File not found: {full_img_path}")  # Inform user about missing file
plt.show()
