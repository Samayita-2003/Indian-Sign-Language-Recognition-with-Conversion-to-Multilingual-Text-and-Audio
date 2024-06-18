import numpy as np
import cv2
import mediapipe as mp
import os
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(max_num_hands=2, static_image_mode=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)

IMAGES_PATH = r'C:\\Users\\KIIT\\Desktop\\College\\6thSem\\MinorProj\\MinorProject\\datasets'

data = []
labels = []

for dir_ in os.listdir(IMAGES_PATH):
    for img_path in os.listdir(os.path.join(IMAGES_PATH, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(IMAGES_PATH, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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

            data.append(data_aux)
            labels.append(dir_)

# Find the maximum length among data samples
max_length = max(len(sample) for sample in data)

# Pad shorter samples with zeros to match the maximum length
padded_data = []
for sample in data:
    if len(sample) < max_length:
        padded_sample = np.pad(sample, (0, max_length - len(sample)), mode='constant')
    else:
        padded_sample = sample[:max_length]  # Truncate longer samples
    padded_data.append(padded_sample)

# Save the padded data and labels to 'data_padded.pickle'
data_dict = {'data': padded_data, 'labels': labels}
with open('data_padded.pickle', 'wb') as f:
    pickle.dump(data_dict, f)




