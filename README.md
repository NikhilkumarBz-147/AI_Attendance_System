# Fresh Smart Attendance System (MySQL)

This is a clean project. Do not mix these files with the previous project.

## 1. Delete/recreate the old database
`database.sql` intentionally starts with:
`DROP DATABASE IF EXISTS smart_attendance;`

Open MySQL Workbench -> new SQL tab -> open/copy `database.sql` -> execute all.

WARNING: this deletes the old smart_attendance database and its data.

## 2. Configure MySQL
Copy `.env.example` to a new file named `.env`.

Edit `.env`:
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=YOUR_REAL_MYSQL_PASSWORD
MYSQL_DATABASE=smart_attendance
SECRET_KEY=anything-long-and-random

Do not put quotes around the password unless the quotes are actually part of the password.

## 3. Create a clean virtual environment (Windows)
Open CMD/PowerShell in this project folder:

python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

## 4. Run
python app.py

First run downloads YuNet and SFace model files, so internet is needed once.

Open:
Student attendance: http://127.0.0.1:5000/attendance
Teacher login: http://127.0.0.1:5000/teacher/login

Teacher:
username: teacher
password: teacher123

## 5. Check DB connection
Open:
http://127.0.0.1:5000/health

Expected:
{"database":"smart_attendance","ok":true}

## 6. Demo order
1. Teacher Login
2. Students -> Add Student
3. Enter UID + Name (there is NO roll number)
4. Capture face
5. Add at least two students
6. Logout
7. Attendance -> face camera
8. First recognition: Attendance marked successfully
9. Same student again: Attendance already marked
10. Teacher login -> Dashboard
11. Dashboard shows Present and Absent students
12. Records -> choose a date
13. Students -> Edit/Delete

## Database tables
teacher_users(id, username, password_hash, created_at)
students(id, student_id, name, embedding, created_at)
attendance(id, student_id, attendance_date, attendance_time, status)

## Important
- This is an academic prototype.
- Face embeddings are stored in MySQL as LONGBLOB.
- One attendance row per student per date is enforced by MySQL UNIQUE(student_id, attendance_date).
- Student-facing pages cannot access teacher management without login.
