import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("face_mask_detector.h5")

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = frame[y:y+h, x:x+w]
        try:
            face_resized = cv2.resize(face, (150, 150))
            normalized = face_resized / 255.0
            reshaped = np.reshape(normalized, (1, 150, 150, 3))

            # Make prediction
            prediction = model.predict(reshaped)[0][0]
            label = " No Mask" if prediction <= 0.5 else " Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except Exception as e:
            print("Error:", e)
            continue

    # Display the frame
    cv2.imshow("Face Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()