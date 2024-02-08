import cv2
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime
import imgaug.augmenters as iaa
import cv2

# Initialize the augmenter
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # 50% horizontal flips
    iaa.Affine(rotate=(-20, 20))  # Rotate images between -20 to 20 degrees
])

# Define the path to the "Training images" directory
base_path = 'Training images'

# Initialize a list to store all augmented images
all_images = []

# Iterate through all images in the "Training images" directory
for img_name in os.listdir(base_path):
    img_path = os.path.join(base_path, img_name)

    # Load the image using OpenCV
    img = cv2.imread(img_path)

    # Check if the image is not None
    if img is not None:
        # Augment the image
        augmented_img = augmenter.augment_image(img)

        all_images.append(augmented_img)

# Function to find and encode faces in a given folder
def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)

        if len(face_encodings) == 0:
            print(f"No faces detected in image")
        else:
            # Assuming there's only one face in each image, so take the first one
            encode = face_encodings[0]
            encodeList.append(encode)

    return encodeList

# Initialize variables
base_path = 'Training images'
classNames = []
encodeListKnown = []

# Traverse all folders in the "Training images" directory
for student_name in os.listdir(base_path):
    student_dir = os.path.join(base_path, student_name)

    if os.path.isdir(student_dir):
        print(f"Processing images for {student_name}")
        student_images = []
        for img_name in os.listdir(student_dir):
            img_path = os.path.join(student_dir, img_name)
            img = cv2.imread(img_path)
            student_images.append(img)

        student_encodings = findEncodings(student_images)

        # Append the encodings and names to the main lists
        if student_encodings:
            encodeListKnown.extend(student_encodings)
            classNames.extend([student_name] * len(student_encodings))

# Save the encodeListKnown into a pickle file
with open("known_faces.pkl", "wb") as f:
    pickle.dump(list(zip(encodeListKnown, classNames)), f)

print("Trained model saved in known_faces.pkl")


#TESTING
# Define the path to the "Testing images" directory
test_base_path = 'Testing images'

# Initialize variables for testing
test_classNames = []
test_encodeListKnown = []

# Initialize a list to store all test images
all_test_images = []

# Iterate through all images in the "Testing images" directory
for img_name in os.listdir(test_base_path):
    img_path = os.path.join(test_base_path, img_name)

    # Load the image using OpenCV
    img = cv2.imread(img_path)

    # Check if the image is not None
    if img is not None:
        all_test_images.append(img)

# Function to find and encode faces in a given list of images
def findTestEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)

        if len(face_encodings) == 0:
            print(f"No faces detected in image")
        else:
            # Assuming there's only one face in each image, so take the first one
            encode = face_encodings[0]
            encodeList.append(encode)

    return encodeList

# Encode faces in test images
test_encodings = findTestEncodings(all_test_images)

# Load the known faces and names from the pickle file
with open("known_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)

# Unzip the known_faces list
known_encodings, known_names = zip(*known_faces)

# Iterate through test encodings to calculate accuracy, recall, and precision
total_faces = 0
correct_recognitions = 0
true_positives = 0
false_negatives = 0
false_positives = 0

for test_encoding in test_encodings:
    total_faces += 1
    matches = face_recognition.compare_faces(known_encodings, test_encoding)

    if any(matches):
        correct_recognitions += 1
        true_positives += 1
    else:
        false_negatives += 1

# Calculate accuracy for each student
for student_name in set(classNames):
    student_encodings = [encode for encode, name in zip(encodeListKnown, classNames) if name == student_name]

    total_faces = 0
    correct_recognitions = 0

    for encoding in student_encodings:
        for encodeFace in encodeListKnown:
            total_faces += 1
            matches = face_recognition.compare_faces([encoding], encodeFace)

            if any(matches):
                correct_recognitions += 1

    if total_faces > 0:
        accuracy = (correct_recognitions / total_faces) * 100
        print(f"Accuracy for {student_name}: {accuracy:.2f}%")

# Calculate overall accuracy
overall_total_faces = 0
overall_correct_recognitions = 0

for encoding in encodeListKnown:
    for encodeFace in encodeListKnown:
        overall_total_faces += 1
        matches = face_recognition.compare_faces([encoding], encodeFace)

        if any(matches):
            overall_correct_recognitions += 1

if overall_total_faces > 0:
    overall_accuracy = (overall_correct_recognitions / overall_total_faces) * 100
    print(f"Overall Model Accuracy: {overall_accuracy:.2f}%")

# Calculate accuracy
accuracy = (correct_recognitions / total_faces) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Calculate recall and precision
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")