import csv
import pickle
import subprocess
import os


import cv2
from flask import Flask, render_template, Response, request, jsonify
from datetime import datetime

app = Flask(__name__, template_folder='Web2', static_folder='Web2')
captured_images_count = 0

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("haarcascade_frontalface_default.xml")

with open("labels.pickles", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(2)
run_face_recognition = False
detected_info = []
unique_names = set() 

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image(name, image_data):

    directory = os.path.join(os.path.dirname(__file__), 'FaceData', name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    existing_images_count = len(os.listdir(directory))

    filename = f"img{existing_images_count + 1}.jpg"
    file_path = os.path.join(directory, filename)

    with open(file_path, 'wb') as file:
        file.write(image_data)

    return file_path

def gen_frames():
    global detected_info, unique_names
    while run_face_recognition:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for x, y, w, h in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id_, conf = recognizer.predict(roi_gray)
            if conf >= 25 and conf <= 85:
                name = labels[id_]
                detection_time = datetime.now()
                detected_info.append({"name": name, "date": detection_time.strftime('%Y-%m-%d'),
                                      "time": detection_time.strftime('%H:%M:%S')})
                if name not in unique_names:
                    unique_names.add(name)
                    detected_info.append({"name": name, "date": detection_time.strftime('%Y-%m-%d'),
                                          "time": detection_time.strftime('%H:%M:%S')})
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/home', methods=["GET"])
def home():
    try:
        return render_template("home.html")
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/register', methods=["GET"])
def register():
    try:
        return render_template("register.html")
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/take_attendance', methods=['GET', 'POST'])
def index():
    global run_face_recognition, unique_names
    if request.method == 'POST':
        run_face_recognition = True
        unique_names = set()  
    return render_template('new.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detected_names')
def get_detected_names():
    return jsonify({'names': detected_info})

@app.route('/save_face', methods=['POST'])
def save_face():
    try:
        name = request.form['name']
        image_data = request.files['imageData'].read()

        file_path = save_image(name, image_data)

        return jsonify({'status': 'success', 'message': 'File saved successfully.', 'file_path': file_path}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/save_attendance')
def save_attendance():
    try:
        global detected_info

        detected_names = set()  
        csv_file_path = 'attendance.csv'

        existing_data = []
        try:
            with open(csv_file_path, 'r', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                existing_data = list(csv_reader)
        except FileNotFoundError:
            pass 

        update_date = True
        if existing_data and len(existing_data[0]) > 1:
            last_saved_date = existing_data[0][-1]
            current_date = datetime.now().strftime('%Y-%m-%d')

            if last_saved_date == current_date:
                update_date = False

        with open(csv_file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            if update_date:
                csv_writer.writerow(['', datetime.now().strftime('%Y-%m-%d')])

            for name_info in detected_info:
                name = name_info['name']

                if name not in detected_names:
                    detected_names.add(name)

                    csv_writer.writerow(['', name])  

        detected_info = []

        return jsonify(
            {'status': 'success', 'message': 'Attendance record saved successfully.', 'file_path': csv_file_path}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/train_model', methods=['GET'])
def train_model():
    try:
        subprocess.run(['python', '2.Face_sample_train.py'])

        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002)
