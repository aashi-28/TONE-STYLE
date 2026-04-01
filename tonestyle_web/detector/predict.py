import cv2
import numpy as np
import tensorflow as tf
import joblib
import math

COLOR_MAP = {
    ("light", "Warm"): ["Peach", "Coral", "Gold"],
    ("light", "Cool"): ["Pink", "Lavender", "Blue"],
    ("medium", "Warm"): ["Orange", "Mustard", "Brown"],
    ("medium", "Cool"): ["Purple", "Teal", "Navy"],
    ("dark", "Warm"): ["Red", "Olive", "Gold"],
    ("dark", "Cool"): ["Royal Blue", "Magenta", "Black"],
}

# Load models
cnn_model = tf.keras.models.load_model("skin_tone_model.keras")
voting_model = joblib.load("voting_model.pkl")

def analyze_image(image_path):
    img = cv2.imread(image_path)

    # Resize face (simple version)
    face = cv2.resize(img, (160,160))

    batch = np.expand_dims(face.astype(np.float32), axis=0)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)

    preds = cnn_model.predict(batch)

    # LAB conversion
    avg_rgb = np.mean(face.reshape(-1,3), axis=0).astype(int)
    lab = cv2.cvtColor(np.uint8([[avg_rgb]]), cv2.COLOR_RGB2LAB)[0][0]

    L, a, b = lab

    ita = math.degrees(math.atan((L - 50)/(b+1e-5)))

    features = [[L, a, b, ita]]
    tone = voting_model.predict(features)[0]

    # Undertone logic
    if b > 15:
        undertone = "Warm"
    elif a > 12:
        undertone = "Cool"
    else:
        undertone = "Neutral"
    colors = COLOR_MAP.get((tone, undertone), ["Black", "White"])
    return tone, undertone, colors    