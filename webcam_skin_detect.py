# webcam_skin_detect.py
import time
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# --------- CONFIG ----------
MODEL_FILE = "skin_tone_model.keras"   # or "skin_tone_model.h5"
CLASS_NAMES = ["dark", "light", "medium"]  # adjust to match your model's class order
TARGET_SIZE = (160, 160)                # must match training size
WEBCAM_INDEX = 0                        # 0 is default built-in webcam; change if needed
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ---------------------------

def load_model_safe(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    # Try to load model; if h5 had custom unknown layers, user should re-save in .keras on Colab
    try:
        model = tf.keras.models.load_model(str(p))
    except Exception as e:
        # helpful message
        raise RuntimeError(
            f"Failed to load model '{path}': {e}\n"
            "If you saved as .h5 in Colab, consider re-saving as .keras and re-download."
        )
    return model

def preprocess_frame(frame_bgr):
    # Convert BGR (OpenCV) to RGB and resize
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, TARGET_SIZE)
    arr = img_resized.astype(np.float32)
    batch = np.expand_dims(arr, axis=0)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    return batch

def main():
    print("Loading model...")
    model = load_model_safe(MODEL_FILE)
    print("Model loaded.")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam. Try another WEBCAM_INDEX.")
        return

    prev_time = time.time()
    save_count = 0

    print("Press 'q' to quit, 's' to save a snapshot (image + predicted label).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            # Optionally: crop center or detect face first. Here we pass the full frame.
            batch = preprocess_frame(frame)
            preds = model.predict(batch)  # shape (1, n_classes) or (1,1) if binary

            # Interpret predictions
            if preds.shape[-1] == 1:  # binary / sigmoid
                prob = float(preds[0][0])
                label = CLASS_NAMES[1] if prob >= 0.5 else CLASS_NAMES[0]
                conf = prob if prob >= 0.5 else 1 - prob
            else:
                idx = int(np.argmax(preds, axis=1)[0])
                label = CLASS_NAMES[idx]
                conf = float(np.max(preds))

            # Overlay text
            text = f"{label} ({conf:.2f})"
            # Draw a semi-transparent rectangle as background for text
            (w, h), _ = cv2.getTextSize(text, FONT, 0.9, 2)
            cv2.rectangle(frame, (10, 10), (10 + w + 12, 10 + h + 12), (0,0,0), -1)
            cv2.putText(frame, text, (16, 10 + h + 2), FONT, 0.9, (0, 255, 0), 2)

            # FPS display
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-10), FONT, 0.6, (200,200,200), 1)

            cv2.imshow("Skin Tone Detector (press q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                # save snapshot
                fname = f"snap_{save_count}_{label}.jpg"
                cv2.imwrite(fname, frame)
                print("Saved", fname)
                save_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
