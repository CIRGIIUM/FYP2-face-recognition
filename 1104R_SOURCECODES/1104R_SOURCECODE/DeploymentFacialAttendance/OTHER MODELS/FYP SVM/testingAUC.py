import os
import numpy as np
import cv2 as cv
import dlib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

base_dir = os.path.dirname(os.path.abspath(__file__))
face_dir = os.path.join(base_dir, "FaceData", "train")

# Initialize the face detector and facial landmarks predictor from dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

x_train = []
y_labels = []
label_id = {}
current_id = 0

for root, dirs, files in os.walk(face_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-")

            fetch = cv.imread(path)
            gray = cv.cvtColor(fetch, cv.COLOR_BGR2GRAY)

            if not label in label_id:
                label_id[label] = current_id
                current_id = current_id + 1

            id_ = label_id[label]

            # Detect faces and facial landmarks using dlib
            faces = face_detector(gray)
            for face in faces:
                landmarks = landmark_predictor(gray, face)

                # Extract features (you may need to customize this part)
                features = np.array([landmark.x for landmark in landmarks.parts()] +
                                    [landmark.y for landmark in landmarks.parts()])

                x_train.append(features)
                y_labels.append(id_)

print(f"Number of unique classes: {len(set(y_labels))}")
print(label_id)

print(y_labels)
print(x_train)
with open("labels.pickles", "wb") as f:
    pickle.dump(label_id, f)

# Train SVM classifier only if there are more than one class
if len(set(y_labels)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_labels, test_size=0.2, random_state=42)

    svm_classifier = SVC(kernel='linear', C=1.0)
    svm_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_classifier.predict(X_test)

    # Transform multi-class to binary class for AUC calculation
    mlb = MultiLabelBinarizer()
    y_test_bin = mlb.fit_transform([[i] for i in y_test])
    y_pred_bin = mlb.transform([[i] for i in y_pred])

    # Calculate overall AUC
    overall_auc = roc_auc_score(y_test_bin, y_pred_bin, average='weighted', multi_class='ovr')
    print(f"Overall AUC: {overall_auc:.2f}")

    # Save the trained SVM classifier
    with open("svm_classifier.pkl", "wb") as svm_model:
        pickle.dump(svm_classifier, svm_model)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
else:
    print("Only one class present. AUC calculation and SVM training skipped.")


