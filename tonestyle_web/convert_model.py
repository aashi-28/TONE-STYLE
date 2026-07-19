import tensorflow as tf

model = tf.keras.models.load_model("skin_tone_model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("skin_tone_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Done - skin_tone_model.tflite created")
