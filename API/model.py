import librosa
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model(r"models/best_model.h5")
encoder = joblib.load(r"models/encoder.h5")

fixed_length = 1000

async def predict(audio, sample_rate):
    # Preprocess the audio the same way you did for training
    spectrogram = librosa.feature.mfcc(y=audio, sr=sample_rate)
    spectrogram = librosa.power_to_db(spectrogram)

    # Pad or truncate the spectrogram to the fixed length
    if spectrogram.shape[1] > fixed_length:
        spectrogram = spectrogram[:, :fixed_length]
    else:
        padding = fixed_length - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), 'constant')

    # Reshape the spectrogram before scaling
    # spectrogram = spectrogram.reshape(-1, 1000)  # Assuming 1000 features
    spectrogram = np.array(spectrogram)

    # Reshape the spectrogram back to its original shape
    spectrogram = spectrogram.reshape(1, 20, 1000, 1)

    # Use the model to make a prediction
    prediction = model.predict(spectrogram)

    # Decode the prediction
    predicted_class = encoder.inverse_transform([np.argmax(prediction)])

    return predicted_class[0]