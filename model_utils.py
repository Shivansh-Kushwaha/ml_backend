import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import joblib
from PIL import Image

# --- CONFIGURATION ---
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

CROPS = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
    'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes',
    'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton',
    'jute', 'coffee'
]

def check_disease_classes(dataset_dir="data/plant_village"):
    if os.path.exists(dataset_dir):
        detected_classes = sorted(os.listdir(dataset_dir))
        if len(detected_classes) > 0 and detected_classes != DISEASE_CLASSES:
            print("WARNING: detected_classes in dataset do not match DISEASE_CLASSES!")
            return detected_classes
    return DISEASE_CLASSES

# DISEASE_CLASSES = check_disease_classes()
def get_disease_model():
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def preprocess_image(image, target_size=(224, 224)):
    """
    Standard preprocessing for EfficientNetB0.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def load_crop_model(model_path):
    """Loads the Scikit-Learn Random Forest model."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def load_disease_model(model_path):
    """Loads the TensorFlow Keras model."""
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            print(f"CRITICAL ERROR: Model loading failed ({e}).")
            return None
            
    print("CRITICAL ERROR: Model file not found.")
    return None
