import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading

# Load the RandomForestClassifier model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

labels_dict = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
               'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H',
               'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P',
               'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X',
               'Y': 'Y', 'Z': 'Z'}

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak_character(character):
    engine.say(character)
    engine.runAndWait()

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            for landmark in hand_landmarks.landmark:
                x = landmark.x - min(x_)
                y = landmark.y - min(y_)
                data_aux.append(x)
                
                
                
                data_aux.append(y)

        # Pad data_aux to a fixed length of 84 features
        padded_data_aux = np.pad(data_aux, (0, 84 - len(data_aux)), mode='constant')

        prediction = model.predict([padded_data_aux])  # Make prediction
        predicted_character = labels_dict[str(prediction[0])]  # Convert prediction to string

        # Draw the predicted character on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size, _ = cv2.getTextSize(predicted_character, font, font_scale, thickness)
        text_x = 10
        text_y = frame.shape[0] - 10
        cv2.putText(frame, predicted_character, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        # Speak the predicted character in a separate thread
        threading.Thread(target=speak_character, args=(predicted_character,)).start()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()