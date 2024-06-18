import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from googletrans import Translator
from gtts import gTTS
import tkinter as tk
from tkinter import ttk

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

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

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def capture_images(n):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    for i in range(n):
        print(f'Preparing to capture image {i + 1} in 5 seconds...')
        time.sleep(2)
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        img_path = os.path.join(data_dir, f'image_{i}.jpg')
        cv2.imwrite(img_path, frame)
        print(f'Image {i + 1} saved.')


        cv2.waitKey(2000)


    cap.release()
    cv2.destroyAllWindows()


    open_gui()

def open_gui():
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

    predictions_str = ''.join(predictions)

    root = tk.Tk()
    root.title("Hand Gesture Recognition")

    predictions_label = tk.Label(root, text="Predictions:")
    predictions_label.pack()

    predictions_text = tk.Text(root, height=2, width=30)
    predictions_text.pack()
    predictions_text.insert(tk.END, predictions_str)

    language_label = tk.Label(root, text="Select language for translation:")
    language_label.pack()

    language_var = tk.StringVar()
    language_dropdown = ttk.Combobox(root, textvariable=language_var, values=["Hindi", "Bengali", "Telugu", "Gujarati"], state="readonly")
    language_dropdown.pack()

    translated_label = tk.Label(root, text="Translated text:")
    translated_label.pack()

    translated_text = tk.Text(root, height=2, width=30)
    translated_text.pack()

    play_audio_button = tk.Button(root, text="Play Audio", state=tk.DISABLED)
    play_audio_button.pack()

    def translate_text(event=None):
        dest_lang = {"Hindi": "hi", "Bengali": "bn", "Telugu": "te", "Gujarati": "gu"}[language_var.get()]
        translator = Translator()
        translated_text_str = translator.translate(predictions_str, dest=dest_lang).text
        translated_text.delete("1.0", tk.END)
        translated_text.insert(tk.END, translated_text_str)

        tts = gTTS(text=translated_text_str, lang=dest_lang)
        audio_file_name = dest_lang + "_audio.mp3"
        tts.save(audio_file_name)
        play_audio_button.config(state=tk.NORMAL, command=lambda: os.system(f"start {audio_file_name}"))

    language_dropdown.bind("<<ComboboxSelected>>", translate_text)

    root.mainloop()

n = int(input('How many pictures do you want to capture? '))

capture_images(n)

