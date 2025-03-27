import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# Load the trained Keras model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "siren_classifier.h5")

try:
    # Register the custom objects
    custom_objects = {
        'DepthwiseConv2D': DepthwiseConv2D
    }
    
    # Try loading with custom objects
    model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

if model:
    print("Model expects input shape:", model.input_shape)
else:
    print("Warning: Model not loaded - using default input shape")

CLASSES = ['Low Priority', 'Medium Priority', 'High Priority']

def generate_spectrogram(audio_path):
    """Generates and saves a spectrogram image from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=4000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Plot and save spectrogram
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=4000, cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram of the Siren")

        # Save spectrogram image
        spectrogram_path = os.path.splitext(audio_path)[0] + "_spectrogram.png"
        plt.savefig(spectrogram_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        return spectrogram_path
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        return None

def extract_features(file_path, max_pad_len=100):
    try:
        if file_path.endswith(".mp3"):
            file_path = convert_mp3_to_wav(file_path)

        audio, sample_rate = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Padding
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        # Reshape to match model input
        mfccs = mfccs.flatten()  # Flatten to 1D
        mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension

        print("Extracted feature shape:", mfccs.shape)  # Debugging

        return mfccs  # Return flattened feature

    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def classify_siren(audio_path):
    """Classifies a siren based on the trained ML model."""
    try:
        if not model:
            return "⚠️ Model Not Loaded - Using Default Classification"

        features = extract_features(audio_path)
        if features is None:
            return "⚠️ Error Processing Audio"
        
        prediction = model.predict(features)
        predicted_class = CLASSES[np.argmax(prediction)]

        return f"🚑 Detected Siren Type: {predicted_class}"
    except Exception as e:
        print(f"Error classifying siren: {e}")
        return "⚠️ Error During Classification"

# Test with an example audio file
if __name__ == "__main__":
    audio_file = "sample_siren.wav"
    
    if os.path.exists(audio_file):
        spectrogram_image = generate_spectrogram(audio_file)
        classification = classify_siren(audio_file)
        print(f"Siren Classification: {classification}")
    else:
        print("Error: Sample siren audio file not found!")
        
