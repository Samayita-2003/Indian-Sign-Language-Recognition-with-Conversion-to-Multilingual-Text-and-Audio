import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from googletrans import Translator
from gtts import gTTS

# Load the RandomForestClassifier model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

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


data_dir = './DataForRecognition'

# Ensure the directory exists, if not create it
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Ask the user how many pictures to capture
n = int(input('How many pictures do you want to capture? '))


# Initialize the webcam
cap = cv2.VideoCapture(0)

# Capture images every 30 seconds
for i in range(n):
    # Notify the user to prepare for the next capture
    print(f'Preparing to capture image {i + 1} in 5 seconds...')  
    time.sleep(2)
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the captured frame
    cv2.imshow('frame', frame)

    # Save the captured frame
    img_path = os.path.join(data_dir, f'image_{i}.jpg')
    cv2.imwrite(img_path, frame)
    print(f'Image {i + 1} saved.')

    # Wait for 1 second before capturing the next image
    cv2.waitKey(2000)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()



# Iterate over the captured images, apply hand landmarks, normalization, padding, and predict the hand gestures
predictions = []
for img_path in os.listdir(data_dir):
    img = cv2.imread(os.path.join(data_dir, img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []
        for hand_landmarks in results.multi_hand_landmarks:
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

        padded_data_aux = np.pad(data_aux, (0, 84 - len(data_aux)), mode='constant')
        prediction = model.predict([padded_data_aux])
        predicted_character = labels_dict[str(prediction[0])]
        predictions.append(predicted_character)
    if len(predictions) >= n:
        break

print("Predictions for captured images:")
predictions_str = ''.join(predictions)
print(predictions_str)

translator = Translator()
print("Select the language to translate the text:")
print("1. Hindi")
print("2. Bengali")
print("3. Telugu")
print("4. Gujarati")
choice = input("Enter your choice (1, 2, 3 or 4): ")

if choice == '1':
    dest_lang = 'hi'
    print("Selected language: Hindi")
elif choice == '2':
    dest_lang = 'bn'
    print("Selected language: Bengali")
elif choice == '3':
    dest_lang = 'te'
    print("Selected language: Telugu")
elif choice == '4':
    dest_lang = 'gu'
    print("Selected language: Gujarati")
else:
    print("Invalid choice. Please enter 1, 2, 3, or 4.")

# Translate the text to the chosen language
translated_text = translator.translate(predictions_str, dest=dest_lang).text
print("Translated text:", translated_text)

# Create a gTTS object for the translated text
tts = gTTS(text=translated_text, lang=dest_lang)

# Save the audio to a file
audio_file_name = dest_lang + "_audio.mp3"
tts.save(audio_file_name)

# Play the audio
os.system(f"start {audio_file_name}")

