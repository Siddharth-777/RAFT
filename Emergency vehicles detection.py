import numpy as np
import tensorflow
import cv2
import librosa
import os
from scipy.fft import fft, fftfreq

frameWidth = 640
frameHeight = 480

# Corrected path for macOS
video_path = "/Users/anjanarangarajan/Desktop/Emergency-vehicles-detection-main/ambulance.mp4"
model_path = "/Users/anjanarangarajan/Desktop/Emergency-vehicles-detection-main/keras_model.h5"
audio_path = "/Users/anjanarangarajan/Desktop/Emergency-vehicles-detection-main/sound_3.wav"

cap = cv2.VideoCapture(video_path)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
np.set_printoptions(suppress=True)

# Check if the model file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please check the path.")

# Load the trained model
model = tensorflow.keras.models.load_model(model_path)

def emergency(image):
    samples, sam_rate = librosa.load(audio_path, sr=None, mono=True, offset=0.0, duration=None)

    def fft_plot(audio, sam_rate):
        n = len(audio)
        T = 1 / sam_rate
        yf = fft(audio)
        xf = fftfreq(n, T)
        val = np.argmax(yf)
        return np.abs(xf[val])

    audio_freq = fft_plot(samples, sam_rate)
    freq = audio_freq.round()
    
    if 700 <= freq <= 1500:
        cv2.putText(image, 'Ambulance Detected', (450, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        print("Emergency vehicle detected!")

amc = 0
while True:
    success, img = cap.read()
    if not success:
        break

    image = img.copy()
    img_resized = cv2.resize(img, (224, 224))
    normalized_image_array = (img_resized.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    ambulance_detected = prediction[0][0].round()
    fire_engine_detected = prediction[0][1].round()
    police_car_detected = prediction[0][2].round()
    traffic_detected = prediction[0][3].round()

    if ambulance_detected == 1:
        amc += 1
    if amc == 20:
        amc = 0
        emergency(image)

    if fire_engine_detected == 1:
        cv2.putText(image, 'Fire Engine Detected', (450, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    if police_car_detected == 1:
        cv2.putText(image, 'Police Car Detected', (450, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    info_text = f'Ambulance: {ambulance_detected} | Fire Engine: {fire_engine_detected} | Police Car: {police_car_detected} | Traffic: {traffic_detected}'
    cv2.putText(image, info_text, (15, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Result", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
