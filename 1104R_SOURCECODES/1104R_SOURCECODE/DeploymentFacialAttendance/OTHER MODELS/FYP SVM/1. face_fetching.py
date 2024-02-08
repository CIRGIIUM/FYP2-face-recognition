import dlib
import cv2 as cv
import os
import numpy as np
from sklearn.model_selection import train_test_split

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img_name = input("Enter name:")
dir_ = r"C:\Users\User\PycharmProjects\DeepFace\FaceData"
names = [input("Enter name:")]

for face in names:
    path = os.path.join(dir_, face)
    if not os.path.exists(path):
        os.mkdir(path)

for folder in names:
    id_ = 0
    user_input = input("Press s to start fetching faces: ").lower()
    if user_input != 's':
        print('Invalid input!')
        exit()
    while id_ < 10:
        _, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        detector = dlib.get_frontal_face_detector()
        faces = detector(gray, 1)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            roi = gray[y:y+h, x:x+w]

            cv.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 90), 5)

            filename = os.path.join(dir_, folder, f'img{id_}.jpg')
            cv.imwrite(filename, roi)
            id_ += 1

        cv.imshow("Feed", frame)

        if cv.waitKey(1) & 0xFF == ord('x'):
            break

print("Faces Collected!")

# Split data into training and testing sets
folder_path = os.path.join(dir_, img_name)
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
labels = [img_name] * len(images)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Save training and testing data to separate directories
train_dir = os.path.join(dir_, 'train')
test_dir = os.path.join(dir_, 'test')

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(test_dir):
    os.mkdir(test_dir)

for i, (img_path, label) in enumerate(zip(X_train, y_train)):
    filename = os.path.join(train_dir, f"{label}_{i}.jpg")
    img = cv.imread(img_path)
    cv.imwrite(filename, img)

for i, (img_path, label) in enumerate(zip(X_test, y_test)):
    filename = os.path.join(test_dir, f"{label}_{i}.jpg")
    img = cv.imread(img_path)
    cv.imwrite(filename, img)

cap.release()
cv.destroyAllWindows()