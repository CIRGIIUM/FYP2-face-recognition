@echo off
set SCRIPT_PATH=%~dp0
set FLASK_APP=%SCRIPT_PATH%\Face_Attendance_app.py
set FLASK_ENV=development
set FLASK_DEBUG=1
start cmd /k "flask run --port=5002"
timeout /t 5
start chrome http://127.0.0.1:5002/home
