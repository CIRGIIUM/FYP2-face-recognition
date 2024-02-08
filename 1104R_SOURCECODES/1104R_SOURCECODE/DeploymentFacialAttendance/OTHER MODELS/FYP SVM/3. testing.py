import cv2 as cv
import dlib
import pickle
import numpy as np
from sklearn.svm import SVC

# Load the trained SVM classifier
with open("svm_classifier.pkl", "rb") as svm_model:
    svm_classifier = pickle.load(svm_model)

# Initialize the face detector and facial landmarks predictor from dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load label mapping
with open("labels.pickles", "rb") as f:
    label_id = pickle.load(f)

# Create inverse label mapping for prediction output
id_label = {v: k for k, v in label_id.items()}

# Open the webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces and facial landmarks using dlib
    faces = face_detector(gray)
    for face in faces:
        landmarks = landmark_predictor(gray, face)

        # Extract features for testing
        features = np.array([landmark.x for landmark in landmarks.parts()] +
                            [landmark.y for landmark in landmarks.parts()])

        # Predict using the trained SVM classifier
        prediction = svm_classifier.predict([features])
        predicted_label = id_label.get(prediction[0], "Unknown")

        # Draw bounding box and label on the frame
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(frame, predicted_label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv.imshow("Face Recognition", frame)

    # Break the loop when 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
