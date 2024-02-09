import librosa
import numpy as np
import tensorflow as tf
import joblib
import io
from PIL import Image
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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



async def Diagnose(image_bytes):


  try: 

    model_path='models/model2.h5'
  #model path
  
    load_model = tf.keras.models.load_model(model_path)
  #image path
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != "RGB":
        image = image.convert("RGB")
        
        # Save as JPG to a bytes buffer
        img_data = io.BytesIO()
        image.save(img_data, format='JPEG')
        image = Image.open(io.BytesIO(img_data.getvalue()))


    image_size_g=(64,64)
    img = image
    img = img.resize(image_size_g)


    aug=ImageDataGenerator(rescale=1./255)
    img = img_to_array(img)
    img = img.reshape((1,) + img.shape)
    aug_img = aug.flow(img,batch_size=1)

    predicted = load_model.predict(aug_img)
    prescent = np.amax(predicted.round(decimals=2))
    predicted_class = np.argmax(predicted.round(decimals=2))

    classes = {0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
                1: ('bcc' , ' basal cell carcinoma'),
                2 :('bkl', 'benign keratosis-like lesions'),
                3: ('df', 'dermatofibroma'),
                4: ('mel', 'melanoma'),
                5: ('nv', ' melanocytic nevi'),
                6: ('vasc', ' pyogenic granulomas and hemorrhage')
                
                }


    return classes[predicted_class][0]
  except Exception as e:
        print(f"Error converting {image_path}: {e}")
        return e 
        
