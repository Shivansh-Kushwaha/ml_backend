import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.applications import MobileNetV3Small
import joblib
from PIL import Image

# --- CONFIGURATION ---
DISEASE_CLASSES = [
    'Apple___Healthy', 'Apple___Rotten', 'Apple___Black_rot', 'Apple___Rust', 'Apple___Scab',

    'Banana___Healthy', 'Banana___Rotten',

    'Bellpepper___Healthy', 'Bellpepper___Rotten',

    'Carrot___Healthy', 'Carrot___Rotten',

    'Cassava___Bacterial_blight', 'Cassava___Brown_streak_disease',
    'Cassava___Green_mottle', 'Cassava___Healthy', 'Cassava___Mosaic_disease',

    'Cherry___Healthy', 'Cherry___Powdery_mildew',

    'Chili___Healthy', 'Chili___Leaf_curl', 'Chili___Leaf_spot',
    'Chili___Whitefly', 'Chili___Yellowish',

    'Coffee___Cercospora_leaf_spot', 'Coffee___Healthy',
    'Coffee___Red_spider_mite', 'Coffee___Rust',

    'Corn___Common_rust', 'Corn___Gray_leaf_spot',
    'Corn___Healthy', 'Corn___Northern_leaf_blight',

    'Cucumber___Healthy', 'Cucumber___Diseased', 'Cucumber___Rotten',

    'Guava___Healthy', 'Guava___Diseased', 'Guava___Rotten',

    'Grape___Healthy', 'Grape___Rotten', 'Grape___Black_measles',
    'Grape___Black_rot', 'Grape___Leaf_blight',

    'Jamun___Healthy', 'Jamun___Diseased',

    'Jujube___Healthy', 'Jujube___Rotten',

    'Lemon___Healthy', 'Lemon___Diseased',

    'Mango___Healthy', 'Mango___Diseased', 'Mango___Rotten',

    'Orange___Healthy', 'Orange___Rotten',

    'Peach___Healthy', 'Peach___Bacterial_spot',

    'Pepper_bell___Healthy', 'Pepper_bell___Bacterial_spot',

    'Pomegranate___Healthy', 'Pomegranate___Diseased', 'Pomegranate___Rotten',

    'Potato___Healthy', 'Potato___Rotten',
    'Potato___Early_blight', 'Potato___Late_blight',

    'Rice___Healthy', 'Rice___Brown_spot', 'Rice___Leaf_blast',
    'Rice___Neck_blast', 'Rice___Hispa',

    'Soybean___Healthy', 'Soybean___Bacterial_blight', 'Soybean___Caterpillar',
    'Soybean___Diabrotica_speciosa', 'Soybean___Downy_mildew',
    'Soybean___Mosaic_virus', 'Soybean___Powdery_mildew',
    'Soybean___Rust', 'Soybean___Southern_blight',

    'Strawberry___Healthy', 'Strawberry___Rotten', 'Strawberry___Leaf_scorch',

    'Sugarcane___Healthy', 'Sugarcane___Bacterial_blight',
    'Sugarcane___Red_rot', 'Sugarcane___Red_stripe', 'Sugarcane___Rust',

    'Tea___Healthy', 'Tea___Algal_leaf', 'Tea___Anthracnose',
    'Tea___Bird_eye_spot', 'Tea___Brown_blight', 'Tea___Red_leaf_spot',

    'Tomato___Healthy', 'Tomato___Rotten',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites',
    'Tomato___Target_spot', 'Tomato___Yellow_leaf_curl_virus',
    'Tomato___Mosaic_virus',

    'Wheat___Healthy', 'Wheat___Brown_rust',
    'Wheat___Yellow_rust', 'Wheat___Septoria'
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

DISEASE_CLASSES = check_disease_classes()

def get_disease_model(num_classes=len(DISEASE_CLASSES)):    """
    Creates a Transfer Learning model based on MobileNetV3Small.
    Why MobileNetV3Small? 
    1. It's optimized for mobile/low-resource environments (high speed).
    2. It uses Neural Architecture Search (NAS) for better accuracy-to-latency ratio.
    3. It's pre-trained on ImageNet, providing a strong feature extraction base.
    """
    base_model = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base layers initially

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), # Reduces spatial dimensions
        layers.Dense(256, activation='relu'), # High-level feature representation
        layers.Dropout(0.3), # Prevents overfitting
        layers.Dense(num_classes, activation='softmax') # Classification head
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image, target_size=(224, 224)):
    """
    Standard preprocessing for MobileNetV3.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    return img_array

def load_crop_model(model_path):
    """Loads the Scikit-Learn Random Forest model."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def load_disease_model(model_path):
    """Loads the TensorFlow Keras model."""
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    return None
