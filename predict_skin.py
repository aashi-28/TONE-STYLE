import tensorflow as tf
import numpy as np
from PIL import Image

# Load the new .keras model (modern format)
model = tf.keras.models.load_model("skin_tone_model.keras")

# Define your class labels (same as in Colab)
class_names = ['dark', 'light', 'medium']

# Load and preprocess test image
img_path = "test1.jpg"
img = Image.open(img_path).convert("RGB").resize((160, 160))
x = np.array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# Predict
pred = model.predict(x)
predicted_class = np.argmax(pred, axis=1)[0]
confidence = np.max(pred)

print(f"Predicted skin tone: {class_names[predicted_class]} (Confidence: {confidence:.2f})")
