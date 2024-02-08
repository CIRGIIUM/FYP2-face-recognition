import cv2
import os

# Create a directory to store training images if it doesn't exist
if not os.path.exists("Training images"):
    os.makedirs("Training images")

# Capture student images
student_name = input("Enter student name: ")
student_dir = os.path.join("Training images", student_name)

# Create a directory for the student if it doesn't exist
if not os.path.exists(student_dir):
    os.makedirs(student_dir)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Capture and save 50 images
for i in range(50):
    ret, frame = cap.read()

    # Display the image
    cv2.imshow("Capture Images", frame)

    # Save the image with a unique name
    image_path = os.path.join(student_dir, f"{student_name}_{i}.jpg")
    cv2.imwrite(image_path, frame)

    # Press 'q' to exit the image capture loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(f"Images of {student_name} captured and saved in the 'Training images' folder.")
