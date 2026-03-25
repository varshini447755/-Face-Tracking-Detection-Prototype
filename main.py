import os

# Check if files are real
for f in ["gender_net.caffemodel", "gender_deploy.prototxt"]:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"{f} found. Size: {size:.2f} MB")
    else:
        print(f"ERROR: {f} is missing from the folder!")
        
import cv2

# --- 1. SETUP MODELS ---
# Load face detector (built into OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your downloaded gender models
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

GENDER_LIST = ['Male', 'Female']
# These are statistical constants the model needs to "see" colors correctly
MODEL_MEAN_VALUES = (78.42, 87.76, 114.89)

# --- 2. START CAMERA ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # Count tracking
    cv2.putText(frame, f"People: {len(faces)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        # Draw the tracking box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # GENDER INFERENCE
        # Crop the face and prep it for the neural network
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        gender_net.setInput(blob)
        preds = gender_net.forward()
        gender = GENDER_LIST[preds[0].argmax()]

        # Display result
        cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Twite AI Prototype', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()