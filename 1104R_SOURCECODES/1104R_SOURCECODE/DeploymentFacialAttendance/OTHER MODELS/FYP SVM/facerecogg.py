import dlib

# Load an image of the person you want to recognize
known_image_path = "C:\\Users\\User\\PycharmProjects\\DeepFace\\Afiefah\\afiefah_1.jpg"
known_image = dlib.load_rgb_image(known_image_path)
known_face_locations = [dlib.rectangle(0, 0, known_image.shape[1], known_image.shape[0])]
known_encoding = dlib.face_recognition_model_v1([known_image], known_face_locations)[0]

# Load an unknown image containing faces
unknown_image_path = "C:\\Users\\User\\PycharmProjects\\DeepFace\\Test\\afie.jpg"
unknown_image = dlib.load_rgb_image(unknown_image_path)

# Find all face detections and encodings in the unknown image
unknown_face_locations = dlib.get_frontal_face_detector()(unknown_image)
unknown_face_encodings = dlib.face_recognition_model_v1([unknown_image], unknown_face_locations)

# Compare each face found in the unknown image to the known face
for unknown_encoding in unknown_face_encodings:
    # Check if the face matches the known face
    distance = dlib.face_distance([known_encoding], unknown_encoding)

    if distance[0] < 0.6:  # Adjust the threshold based on your scenario
        print("Face recognized!")
    else:
        print("Face not recognized.")

