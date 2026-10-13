import csv
from io import StringIO

import os
import urllib.request
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path

import cv2
import mysql.connector
import numpy as np
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent

load_dotenv(BASE / ".env")

app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "dev-secret-change-me"
)

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv(
        "MYSQL_DATABASE",
        "smart_attendance"
    ),
}


# ============================================================
# FACE RECOGNITION MODEL CONFIGURATION
# ============================================================

MODELS = BASE / "models"

YUNET = MODELS / (
    "face_detection_yunet_2023mar.onnx"
)

SFACE = MODELS / (
    "face_recognition_sface_2021dec.onnx"
)

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)

SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)

# Cosine similarity threshold
MATCH_THRESHOLD = 0.363


# ============================================================
# DATABASE
# ============================================================

def get_db():
    """
    Create and return a MySQL database connection.
    """
    return mysql.connector.connect(**DB_CONFIG)


# ============================================================
# TEACHER ACCOUNT
# ============================================================

def ensure_teacher():
    """
    Creates the default teacher account if it doesn't exist.

    Username:
        teacher

    Password:
        teacher123

    The password is stored as a secure hash.
    """

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id
        FROM teacher_users
        WHERE username=%s
        """,
        ("teacher",)
    )

    existing = cur.fetchone()

    if not existing:
        password_hash = generate_password_hash(
            "teacher123"
        )

        cur.execute(
            """
            INSERT INTO teacher_users
                (username, password_hash)
            VALUES
                (%s, %s)
            """,
            (
                "teacher",
                password_hash
            )
        )

        conn.commit()

    cur.close()
    conn.close()


# ============================================================
# DOWNLOAD FACE MODELS
# ============================================================

def download_models():
    """
    Download YuNet and SFace models if they don't exist.
    """

    MODELS.mkdir(exist_ok=True)

    models = [
        (YUNET, YUNET_URL),
        (SFACE, SFACE_URL)
    ]

    for path, url in models:

        if (
            not path.exists()
            or path.stat().st_size < 1_000_000
        ):
            print(
                f"Downloading {path.name}..."
            )

            urllib.request.urlretrieve(
                url,
                path
            )


# ============================================================
# FACE DETECTOR
# ============================================================

def detector():

    return cv2.FaceDetectorYN.create(
        str(YUNET),
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )


# ============================================================
# FACE RECOGNIZER
# ============================================================

def recognizer():

    return cv2.FaceRecognizerSF.create(
        str(SFACE),
        ""
    )


# ============================================================
# FACE DETECTION
# ============================================================

def detect(det, frame):

    det.setInputSize(
        (
            frame.shape[1],
            frame.shape[0]
        )
    )

    _, faces = det.detect(frame)

    if faces is None:
        return np.empty(
            (0, 15),
            dtype=np.float32
        )

    return faces


# ============================================================
# CREATE FACE EMBEDDING
# ============================================================

def feature(rec, frame, face):

    aligned = rec.alignCrop(
        frame,
        face
    )

    return rec.feature(
        aligned
    ).astype(np.float32)


# ============================================================
# TEACHER AUTHORIZATION DECORATOR
# ============================================================

def teacher_required(fn):

    @wraps(fn)
    def wrapped(*args, **kwargs):

        if not session.get("teacher_id"):
            return redirect(
                url_for("teacher_login")
            )

        return fn(*args, **kwargs)

    return wrapped


# ============================================================
# GET ALL STUDENTS
# ============================================================

def all_students():

    conn = get_db()

    cur = conn.cursor(
        dictionary=True
    )

    cur.execute(
        """
        SELECT
            id,
            student_id,
            name,
            embedding,
            created_at
        FROM students
        ORDER BY name
        """
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


# ============================================================
# GET ATTENDANCE FOR A PARTICULAR DAY
# ============================================================

def attendance_for_day(day):

    conn = get_db()

    cur = conn.cursor(
        dictionary=True
    )

    cur.execute(
        """
        SELECT
            s.id,
            s.student_id,
            s.name,
            COALESCE(
                a.status,
                'Absent'
            ) AS status,
            a.attendance_time

        FROM students s

        LEFT JOIN attendance a
            ON a.student_id = s.id
            AND a.attendance_date = %s

        ORDER BY s.name
        """,
        (day,)
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


# ============================================================
# PARSE FLOAT
# ============================================================

def parse_float(value):
    """Return a float or None for an empty/invalid value."""
    if value in (None, ""):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_percentage_filter(value, field_name):
    """
    Validate an attendance percentage filter.

    Returns:
        (number, error_message)
    """
    raw = (value or "").strip()

    if raw == "":
        return None, None

    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None, f"{field_name} must be a valid number."

    if not np.isfinite(number):
        return None, f"{field_name} must be a valid number."

    if number < 0 or number > 100:
        return None, f"{field_name} must be between 0 and 100."

    return number, None


# ============================================================
# STUDENT ATTENDANCE STATISTICS
# ============================================================

def attendance_stats_for_student(
    student_id,
    month=None
):

    conn = get_db()

    cur = conn.cursor(
        dictionary=True
    )

    if month:

        cur.execute(
            """
            SELECT
                COUNT(
                    DISTINCT attendance_date
                ) AS present_days,

                MIN(attendance_date)
                    AS first_date,

                MAX(attendance_date)
                    AS last_date

            FROM attendance

            WHERE student_id=%s

            AND attendance_date >= %s

            AND attendance_date <
                DATE_ADD(
                    %s,
                    INTERVAL 1 MONTH
                )
            """,
            (
                student_id,
                month,
                month
            )
        )

    else:

        cur.execute(
            """
            SELECT

                COUNT(
                    DISTINCT attendance_date
                ) AS present_days,

                MIN(attendance_date)
                    AS first_date,

                MAX(attendance_date)
                    AS last_date

            FROM attendance

            WHERE student_id=%s
            """,
            (student_id,)
        )

    result = (
        cur.fetchone()
        or {}
    )

    cur.close()
    conn.close()

    return result


# ============================================================
# MONTHLY STUDENT STATISTICS
# ============================================================

def monthly_student_stats(
    student_id,
    months=12
):
    """
    Return the student's attendance percentage for the last N months.

    The denominator is the number of distinct attendance dates recorded
    for the whole class in each month, not the number of calendar days.
    This keeps the percentage consistent with the Analytics page.
    """

    current_month = date.today().replace(day=1)

    month_starts = []
    year = current_month.year
    month = current_month.month

    for _ in range(months):
        month_starts.append(date(year, month, 1))
        month -= 1

        if month == 0:
            month = 12
            year -= 1

    start = month_starts[-1]

    if current_month.month == 12:
        end = date(current_month.year + 1, 1, 1)
    else:
        end = date(
            current_month.year,
            current_month.month + 1,
            1
        )

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    # Student's present days by month.
    cur.execute(
        """
        SELECT
            DATE_FORMAT(
                attendance_date,
                '%%Y-%%m'
            ) AS month,
            COUNT(DISTINCT attendance_date) AS present_days
        FROM attendance
        WHERE student_id=%s
          AND attendance_date >= %s
          AND attendance_date < %s
          AND status='Present'
        GROUP BY DATE_FORMAT(
            attendance_date,
            '%%Y-%%m'
        )
        """,
        (student_id, start, end)
    )
    student_rows = cur.fetchall()

    # Number of actual attendance-recorded/class days by month.
    cur.execute(
        """
        SELECT
            DATE_FORMAT(
                attendance_date,
                '%%Y-%%m'
            ) AS month,
            COUNT(DISTINCT attendance_date) AS recorded_days
        FROM attendance
        WHERE attendance_date >= %s
          AND attendance_date < %s
        GROUP BY DATE_FORMAT(
            attendance_date,
            '%%Y-%%m'
        )
        """,
        (start, end)
    )
    recorded_rows = cur.fetchall()

    cur.close()
    conn.close()

    present_by_month = {
        row["month"]: int(row["present_days"] or 0)
        for row in student_rows
    }

    recorded_by_month = {
        row["month"]: int(row["recorded_days"] or 0)
        for row in recorded_rows
    }

    output = []

    # Oldest -> newest for the chart.
    for month_start in reversed(month_starts):
        key = month_start.strftime("%Y-%m")
        present = present_by_month.get(key, 0)
        recorded_days = recorded_by_month.get(key, 0)

        percentage = (
            round(present * 100 / recorded_days, 1)
            if recorded_days
            else 0.0
        )

        output.append({
            "month": key,
            "present_days": present,
            "recorded_days": recorded_days,
            "percentage": percentage
        })

    return output


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return redirect(
        url_for(
            "student_attendance"
        )
    )


# ============================================================
# STUDENT ATTENDANCE PAGE
# ============================================================

@app.get("/attendance")
def student_attendance():

    return render_template(
        "student_attendance.html"
    )


# ============================================================
# FACE RECOGNITION + ATTENDANCE API
# ============================================================

@app.post("/api/recognize")
def api_recognize():

    f = request.files.get(
        "frame"
    )

    if not f:

        return jsonify(
            ok=False,
            state="error",
            message=
                "No camera frame received.",
            faces=[]
        ), 400

    frame = cv2.imdecode(
        np.frombuffer(
            f.read(),
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return jsonify(
            ok=False,
            state="error",
            message=
                "Invalid camera frame.",
            faces=[]
        ), 400

    det = detector()
    rec = recognizer()

    faces = detect(
        det,
        frame
    )

    # --------------------------------------------------------
    # NO FACE
    # --------------------------------------------------------

    if len(faces) == 0:

        return jsonify(
            ok=True,
            state="no_face",
            message=
                "No face detected. "
                "Please face the camera.",
            faces=[]
        )

    # --------------------------------------------------------
    # MULTIPLE FACES
    # --------------------------------------------------------

    if len(faces) > 1:

        boxes = []

        for face in faces:

            x, y, w, h = map(
                int,
                face[:4]
            )

            boxes.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "state":
                        "multiple"
                }
            )

        return jsonify(
            ok=True,
            state="multiple",
            message=
                "Please show only "
                "one student.",
            faces=boxes
        )

    # --------------------------------------------------------
    # SINGLE FACE
    # --------------------------------------------------------

    face = faces[0]

    x, y, w, h = map(
        int,
        face[:4]
    )

    emb = feature(
        rec,
        frame,
        face
    )

    students_data = (
        all_students()
    )

    best = None
    best_score = -1.0

    # --------------------------------------------------------
    # FIND BEST MATCH
    # --------------------------------------------------------

    for student in students_data:

        known = np.frombuffer(
            student["embedding"],
            dtype=np.float32
        ).reshape(
            1,
            -1
        )

        score = float(
            rec.match(
                emb,
                known,
                cv2.FaceRecognizerSF_FR_COSINE
            )
        )

        if score > best_score:

            best = student
            best_score = score

    # --------------------------------------------------------
    # UNKNOWN STUDENT
    # --------------------------------------------------------

    if (
        best is None
        or
        best_score < MATCH_THRESHOLD
    ):

        return jsonify(

            ok=True,

            state="unknown",

            name="Unknown Student",

            message=
                "Student is not registered. "
                "Attendance not marked.",

            confidence=
                round(
                    max(
                        best_score,
                        0
                    ) * 100,
                    1
                ),

            faces=[
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "state":
                        "unknown"
                }
            ]
        )

    # --------------------------------------------------------
    # CHECK IF ATTENDANCE ALREADY EXISTS
    # --------------------------------------------------------

    today = date.today()

    conn = get_db()

    cur = conn.cursor(
        dictionary=True
    )

    cur.execute(
        """
        SELECT attendance_time

        FROM attendance

        WHERE student_id=%s

        AND attendance_date=%s
        """,
        (
            best["id"],
            today
        )
    )

    old = cur.fetchone()

    if old:

        attendance_time = str(
            old["attendance_time"]
        )

        cur.close()
        conn.close()

        return jsonify(

            ok=True,

            state="already",

            name=best["name"],

            uid=best["student_id"],

            confidence=
                round(
                    best_score * 100,
                    1
                ),

            message=
                "Attendance already "
                f"marked today at "
                f"{attendance_time}.",

            faces=[
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "state":
                        "already"
                }
            ]
        )

    # --------------------------------------------------------
    # MARK ATTENDANCE
    # --------------------------------------------------------

    now = datetime.now()

    cur.execute(
        """
        INSERT INTO attendance
        (
            student_id,
            attendance_date,
            attendance_time,
            status
        )

        VALUES
        (
            %s,
            %s,
            %s,
            'Present'
        )
        """,
        (
            best["id"],
            today,
            now.time().replace(
                microsecond=0
            )
        )
    )

    conn.commit()

    cur.close()
    conn.close()

    return jsonify(

        ok=True,

        state="marked",

        name=best["name"],

        uid=best["student_id"],

        confidence=
            round(
                best_score * 100,
                1
            ),

        message=
            "Attendance marked "
            "successfully. Present.",

        faces=[
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "state":
                    "marked"
            }
        ]
    )


# ============================================================
# TEACHER LOGIN
# ============================================================

@app.route(
    "/teacher/login",
    methods=["GET", "POST"]
)
def teacher_login():

    if request.method == "POST":

        username = (
            request.form
            .get(
                "username",
                ""
            )
            .strip()
        )

        password = (
            request.form
            .get(
                "password",
                ""
            )
        )

        conn = get_db()

        cur = conn.cursor(
            dictionary=True
        )

        cur.execute(
            """
            SELECT
                id,
                username,
                password_hash

            FROM teacher_users

            WHERE username=%s
            """,
            (username,)
        )

        teacher = cur.fetchone()

        cur.close()
        conn.close()

        if (
            teacher
            and
            check_password_hash(
                teacher["password_hash"],
                password
            )
        ):

            session.clear()

            session["teacher_id"] = (
                teacher["id"]
            )

            session["teacher_username"] = (
                teacher["username"]
            )

            return redirect(
                url_for(
                    "dashboard"
                )
            )

        flash(
            "Invalid username or password.",
            "error"
        )

    return render_template(
        "login.html"
    )


# ============================================================
# TEACHER LOGOUT
# ============================================================

@app.get("/teacher/logout")
def teacher_logout():

    session.clear()

    return redirect(
        url_for(
            "teacher_login"
        )
    )


# ============================================================
# TEACHER DASHBOARD
# ============================================================

@app.get("/dashboard")
@teacher_required
def dashboard():

    today = date.today()
    conn = get_db()
    cur = conn.cursor(dictionary=True)

    # Total students
    cur.execute("""
        SELECT COUNT(*) AS total
        FROM students
    """)
    total_students = int(cur.fetchone()["total"] or 0)

    # Present today
    cur.execute("""
        SELECT COUNT(DISTINCT student_id) AS present
        FROM attendance
        WHERE attendance_date=%s
        AND status='Present'
    """, (today,))
    present_today = int(cur.fetchone()["present"] or 0)

    # Absent today
    absent_today = max(total_students - present_today, 0)

    # Attendance rate
    attendance_rate = (
        round(present_today * 100 / total_students, 1)
        if total_students else 0
    )

    # Today's present attendance
    cur.execute("""
        SELECT
            s.student_id,
            s.name,
            a.status,
            a.attendance_time
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        WHERE a.attendance_date=%s
        ORDER BY a.attendance_time DESC
    """, (today,))
    rows = cur.fetchall()

    # 7-day attendance trend
    start_date = today - timedelta(days=6)

    cur.execute("""
        SELECT
            attendance_date,
            COUNT(DISTINCT student_id) AS present
        FROM attendance
        WHERE attendance_date >= %s
        AND attendance_date <= %s
        AND status='Present'
        GROUP BY attendance_date
        ORDER BY attendance_date
    """, (start_date, today))

    trend_rows = cur.fetchall()

    trend_map = {
        row["attendance_date"]: int(row["present"] or 0)
        for row in trend_rows
    }

    trend_labels = []
    trend_values = []
    current_day = start_date

    while current_day <= today:
        trend_labels.append(current_day.strftime("%d %b"))

        day_present = trend_map.get(current_day, 0)

        day_rate = (
            round(day_present * 100 / total_students, 1)
            if total_students else 0
        )

        trend_values.append(day_rate)
        current_day += timedelta(days=1)

    # Total recorded attendance days
    cur.execute("""
        SELECT COUNT(DISTINCT attendance_date) AS total_days
        FROM attendance
    """)

    total_recorded_days = int(
        cur.fetchone()["total_days"] or 0
    )

    # Students requiring attention
    cur.execute("""
        SELECT
            s.id,
            s.student_id,
            s.name,

            COUNT(
                DISTINCT CASE
                    WHEN a.status='Present'
                    THEN a.attendance_date
                END
            ) AS present_days,

            MAX(a.attendance_date) AS last_attendance

        FROM students s

        LEFT JOIN attendance a
            ON a.student_id = s.id

        GROUP BY
            s.id,
            s.student_id,
            s.name

        ORDER BY
            present_days ASC,
            s.name ASC
    """)

    student_stats = cur.fetchall()
    at_risk_students = []

    for student in student_stats:

        present_days = int(
            student["present_days"] or 0
        )

        if total_recorded_days == 0:
            continue

        percentage = round(
            present_days * 100 / total_recorded_days,
            1
        )

        if percentage < 75:
            at_risk_students.append({
                "id": student["id"],
                "student_id": student["student_id"],
                "name": student["name"],
                "present_days": present_days,
                "attendance_pct": percentage,
                "last_attendance": student["last_attendance"]
            })

    cur.close()
    conn.close()

    return render_template(
        "dashboard.html",
        today=today,

        total=total_students,
        present=present_today,
        absent=absent_today,
        rate=attendance_rate,

        # Backward-compatible names
        total_students=total_students,
        present_today=present_today,
        absent_today=absent_today,
        attendance_rate=attendance_rate,

        rows=rows,

        trend_labels=trend_labels,
        trend_values=trend_values,

        at_risk_students=at_risk_students
    )


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

@app.get("/analytics")
@teacher_required
def analytics():

    # --------------------------------------------------------
    # FILTER VALUES
    # --------------------------------------------------------

    selected_month = (
        request.args.get(
            "month"
        )
        or
        date.today().strftime(
            "%Y-%m"
        )
    )

    search = (
        request.args.get(
            "q",
            ""
        )
        .strip()
    )

    min_pct, min_error = parse_percentage_filter(
        request.args.get("min_pct"),
        "Minimum attendance"
    )

    max_pct, max_error = parse_percentage_filter(
        request.args.get("max_pct"),
        "Maximum attendance"
    )

    if min_error or max_error:
        flash(min_error or max_error, "error")
        return redirect(url_for("analytics"))

    if (
        min_pct is not None
        and max_pct is not None
        and min_pct > max_pct
    ):
        flash(
            "Minimum attendance cannot be greater than maximum attendance.",
            "error"
        )
        return redirect(url_for("analytics"))

    activity = (
        request.args.get(
            "activity",
            "all"
        )
        .strip()
        .lower()
    )

    if activity not in {
        "all",
        "active",
        "inactive",
        "below75"
    }:
        activity = "all"

    # --------------------------------------------------------
    # PARSE MONTH
    # --------------------------------------------------------

    try:

        month_start = datetime.strptime(
            selected_month + "-01",
            "%Y-%m-%d"
        ).date()

    except ValueError:

        month_start = (
            date.today().replace(
                day=1
            )
        )

        selected_month = (
            month_start.strftime(
                "%Y-%m"
            )
        )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    conn = get_db()

    cur = conn.cursor()

    # Total number of distinct attendance dates
    cur.execute(
        """
        SELECT COUNT(
            DISTINCT attendance_date
        )

        FROM attendance
        """
    )

    total_recorded_days = int(
        cur.fetchone()[0]
        or 0
    )

    # Total recorded attendance days
    # for selected month
    cur.execute(
        """
        SELECT COUNT(
            DISTINCT attendance_date
        )

        FROM attendance

        WHERE attendance_date >= %s

        AND attendance_date <
            DATE_ADD(
                %s,
                INTERVAL 1 MONTH
            )
        """,
        (
            month_start,
            month_start
        )
    )

    month_recorded_days = int(
        cur.fetchone()[0]
        or 0
    )

    cur.close()
    conn.close()

    # --------------------------------------------------------
    # ALL STUDENTS
    # --------------------------------------------------------

    students_data = all_students()

    # --------------------------------------------------------
    # OVERALL STATISTICS
    # --------------------------------------------------------

    conn = get_db()

    cur = conn.cursor(
        dictionary=True
    )

    cur.execute(
        """
        SELECT

            s.id,

            COUNT(
                DISTINCT a.attendance_date
            ) AS present_days,

            MAX(
                a.attendance_date
            ) AS last_attendance,

            MIN(
                a.attendance_date
            ) AS first_attendance

        FROM students s

        LEFT JOIN attendance a

            ON a.student_id=s.id

        GROUP BY s.id
        """
    )

    overall_rows = {
        int(row["id"]):
            row
        for row in cur.fetchall()
    }

    # --------------------------------------------------------
    # MONTHLY STATISTICS
    # --------------------------------------------------------

    cur.execute(
        """
        SELECT

            student_id,

            COUNT(
                DISTINCT attendance_date
            ) AS present_days

        FROM attendance

        WHERE attendance_date >= %s

        AND attendance_date <
            DATE_ADD(
                %s,
                INTERVAL 1 MONTH
            )

        GROUP BY student_id
        """,
        (
            month_start,
            month_start
        )
    )

    month_rows = {
        int(row["student_id"]):
            int(
                row["present_days"]
                or 0
            )
        for row in cur.fetchall()
    }

    cur.close()
    conn.close()

    # --------------------------------------------------------
    # CALCULATE STUDENT METRICS
    # --------------------------------------------------------

    for student in students_data:

        sid = int(
            student["id"]
        )

        overall = (
            overall_rows.get(
                sid,
                {}
            )
        )

        present_days = int(
            overall.get(
                "present_days",
                0
            )
            or 0
        )

        month_present = (
            month_rows.get(
                sid,
                0
            )
        )

        last_attendance = (
            overall.get(
                "last_attendance"
            )
        )

        # Overall percentage
        student["overall_pct"] = (

            round(
                present_days *
                100 /
                total_recorded_days,
                1
            )

            if total_recorded_days
            else 0.0
        )

        # Monthly percentage
        student["month_pct"] = (

            round(
                month_present *
                100 /
                month_recorded_days,
                1
            )

            if month_recorded_days
            else 0.0
        )

        student["present_days"] = (
            present_days
        )

        student["month_present"] = (
            month_present
        )

        student["last_attendance"] = (
            last_attendance
        )

        # Active if attended
        # during last 7 days
        student["active"] = bool(

            last_attendance

            and

            (
                date.today()
                -
                last_attendance
            ).days <= 7
        )

        # Risk category
        if student["overall_pct"] < 75:

            student["risk"] = (
                "warning"
            )

        else:

            student["risk"] = (
                "good"
            )

    # --------------------------------------------------------
    # SUMMARY CARDS
    # --------------------------------------------------------

    all_students_count = (
        len(students_data)
    )

    below_75 = sum(
        1
        for student
        in students_data

        if student["overall_pct"]
        < 75
    )

    active_students = sum(
        1
        for student
        in students_data

        if student["active"]
    )

    inactive_students = (
        all_students_count
        -
        active_students
    )

    average_pct = (

        round(
            sum(
                student["overall_pct"]
                for student
                in students_data
            )
            /
            all_students_count,
            1
        )

        if all_students_count

        else 0.0
    )

    # --------------------------------------------------------
    # FILTER STUDENTS
    # --------------------------------------------------------

    filtered = []

    for student in students_data:

        # Search
        search_match = (

            not search

            or

            search.lower()
            in student["name"].lower()

            or

            search.lower()
            in student["student_id"].lower()
        )

        # Minimum %
        min_match = (

            min_pct is None

            or

            student["overall_pct"]
            >= min_pct
        )

        # Maximum %
        max_match = (

            max_pct is None

            or

            student["overall_pct"]
            <= max_pct
        )

        # Activity filter
        activity_match = (

            activity == "all"

            or

            (
                activity == "active"
                and
                student["active"]
            )

            or

            (
                activity == "inactive"
                and
                not student["active"]
            )

            or

            (
                activity == "below75"
                and
                student["overall_pct"]
                < 75
            )
        )

        if (
            search_match
            and
            min_match
            and
            max_match
            and
            activity_match
        ):

            filtered.append(
                student
            )

    # Lowest attendance first
    filtered.sort(
        key=lambda x:
            x["overall_pct"]
    )

    # Chart data
    chart_students = (
        filtered[:15]
    )

    chart_labels = [
        student["name"]
        for student
        in chart_students
    ]

    chart_values = [
        student["overall_pct"]
        for student
        in chart_students
    ]

    # --------------------------------------------------------
    # RENDER ANALYTICS
    # --------------------------------------------------------

    return render_template(

        "analytics.html",

        students=filtered,

        summary={
            "all_students_count": all_students_count,
            "below_75": below_75,
            "active_students": active_students,
            "inactive_students": inactive_students,
            "average_pct": average_pct
        },

        selected_month=
            selected_month,

        search=
            search,

        min_pct=
            min_pct
            if min_pct is not None
            else "",

        max_pct=
            max_pct
            if max_pct is not None
            else "",

        activity=
            activity,

        recorded_days=
            total_recorded_days,

        month_recorded_days=
            month_recorded_days,

        chart_labels=
            chart_labels,

        chart_values=
            chart_values
    )


# ============================================================
# INDIVIDUAL STUDENT ANALYTICS
# ============================================================

@app.get(
    "/students/<int:sid>/analytics"
)
@teacher_required
def student_analytics(sid):

    # --------------------------------------------------------
    # STUDENT
    # --------------------------------------------------------

    conn = get_db()

    cur = conn.cursor(
        dictionary=True
    )

    cur.execute(
        """
        SELECT
            id,
            student_id,
            name,
            created_at

        FROM students

        WHERE id=%s
        """,
        (sid,)
    )

    student = cur.fetchone()

    if not student:

        cur.close()
        conn.close()

        return (
            "Student not found",
            404
        )

    # --------------------------------------------------------
    # ATTENDANCE HISTORY
    # --------------------------------------------------------

    cur.execute(
        """
        SELECT

            attendance_date,
            attendance_time,
            status

        FROM attendance

        WHERE student_id=%s

        ORDER BY
            attendance_date DESC,
            attendance_time DESC
        """,
        (sid,)
    )

    attendance_rows = (
        cur.fetchall()
    )

    cur.close()
    conn.close()

    # --------------------------------------------------------
    # TOTAL RECORDED DAYS
    # --------------------------------------------------------

    conn = get_db()

    cur = conn.cursor()

    cur.execute(
        """
        SELECT COUNT(
            DISTINCT attendance_date
        )

        FROM attendance
        """
    )

    total_recorded_days = int(
        cur.fetchone()[0]
        or 0
    )

    cur.close()
    conn.close()

    # --------------------------------------------------------
    # STUDENT PRESENT DAYS
    # --------------------------------------------------------

    present = sum(

        1

        for row
        in attendance_rows

        if row["status"]
        == "Present"
    )

    # --------------------------------------------------------
    # ATTENDANCE %
    # --------------------------------------------------------

    percentage = (

        round(
            present *
            100 /
            total_recorded_days,
            1
        )

        if total_recorded_days

        else 0.0
    )

    # --------------------------------------------------------
    # LAST ATTENDANCE
    # --------------------------------------------------------

    last_attendance = (

        attendance_rows[0]
        ["attendance_date"]

        if attendance_rows

        else None
    )

    # --------------------------------------------------------
    # ACTIVE STATUS
    # --------------------------------------------------------

    active = bool(

        last_attendance

        and

        (
            date.today()
            -
            last_attendance
        ).days <= 7
    )

    # --------------------------------------------------------
    # MONTHLY ANALYTICS
    # --------------------------------------------------------

    monthly = (
        monthly_student_stats(
            sid,
            12
        )
    )

    trend_labels = [
        row["month"]
        for row
        in monthly
    ]

    trend_values = [
        row["percentage"]
        for row
        in monthly
    ]

    # --------------------------------------------------------
    # RENDER
    # --------------------------------------------------------

    return render_template(

        "student_analytics.html",

        student=student,

        attendance_history=
            attendance_rows,

        present_days=
            present,

        total_days=
            total_recorded_days,

        attendance_pct=
            percentage,

        active=
            active,

        last_attendance=
            last_attendance,

        monthly=
            monthly,

        monthly_labels=
            trend_labels,

        monthly_values=
            trend_values
    )


# ============================================================
# STUDENTS PAGE
# ============================================================

@app.get("/students")
@teacher_required
def students():

    return render_template(
        "students.html",
        students=all_students()
    )


# ============================================================
# ADD STUDENT
# ============================================================

@app.route(
    "/students/add",
    methods=["GET", "POST"]
)
@teacher_required
def add_student():

    if request.method == "POST":

        uid = (
            request.form
            .get(
                "student_id",
                ""
            )
            .strip()
        )

        name = (
            request.form
            .get(
                "name",
                ""
            )
            .strip()
        )

        if not uid or not name:

            flash(
                "UID and name are required.",
                "error"
            )

        elif len(uid) > 50:

            flash(
                "Student UID must be 50 characters or fewer.",
                "error"
            )

        elif len(name) > 100:

            flash(
                "Student name must be 100 characters or fewer.",
                "error"
            )

        else:

            return render_template(
                "capture.html",
                uid=uid,
                name=name
            )

    return render_template(
        "add_student.html"
    )


# ============================================================
# REGISTER STUDENT API
# ============================================================

@app.post(
    "/api/students/register"
)
@teacher_required
def register_student_api():

    uid = (
        request.form
        .get(
            "student_id",
            ""
        )
        .strip()
    )

    name = (
        request.form
        .get(
            "name",
            ""
        )
        .strip()
    )

    f = request.files.get(
        "frame"
    )

    if not uid or not name or not f:

        return jsonify(
            ok=False,
            message=
                "UID, name and face image "
                "are required."
        ), 400

    if len(uid) > 50:
        return jsonify(
            ok=False,
            message="Student UID must be 50 characters or fewer."
        ), 400

    if len(name) > 100:
        return jsonify(
            ok=False,
            message="Student name must be 100 characters or fewer."
        ), 400

    frame = cv2.imdecode(
        np.frombuffer(
            f.read(),
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return jsonify(
            ok=False,
            message=
                "Invalid image."
        ), 400

    faces = detect(
        detector(),
        frame
    )

    if len(faces) != 1:

        return jsonify(
            ok=False,
            message=
                "Please show exactly "
                "one face. "
                f"Detected: {len(faces)}"
        ), 400

    emb = feature(
        recognizer(),
        frame,
        faces[0]
    )

    conn = get_db()

    cur = conn.cursor()

    try:

        cur.execute(
            """
            INSERT INTO students
            (
                student_id,
                name,
                embedding
            )

            VALUES
            (
                %s,
                %s,
                %s
            )
            """,
            (
                uid,
                name,
                emb.tobytes()
            )
        )

        conn.commit()

    except mysql.connector.IntegrityError:

        conn.rollback()

        cur.close()
        conn.close()

        return jsonify(
            ok=False,
            message=
                "That UID already exists."
        ), 409

    cur.close()
    conn.close()

    return jsonify(
        ok=True,
        message=
            "Student registered successfully."
    )


# ============================================================
# EDIT STUDENT
# ============================================================

@app.route(
    "/students/<int:sid>/edit",
    methods=["GET", "POST"]
)
@teacher_required
def edit_student(sid):

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    if request.method == "POST":

        uid = request.form.get(
            "student_id",
            ""
        ).strip()

        name = request.form.get(
            "name",
            ""
        ).strip()

        if not uid or not name:
            flash(
                "UID and name are required.",
                "error"
            )

        elif len(uid) > 50:
            flash(
                "Student UID must be 50 characters or fewer.",
                "error"
            )

        elif len(name) > 100:
            flash(
                "Student name must be 100 characters or fewer.",
                "error"
            )

        else:
            try:
                cur.execute(
                    """
                    UPDATE students
                    SET student_id=%s,
                        name=%s
                    WHERE id=%s
                    """,
                    (uid, name, sid)
                )

                if cur.rowcount == 0:
                    conn.rollback()
                    cur.close()
                    conn.close()
                    return "Student not found", 404

                conn.commit()

                cur.close()
                conn.close()

                flash(
                    "Student updated.",
                    "success"
                )

                return redirect(
                    url_for("students")
                )

            except mysql.connector.IntegrityError:
                conn.rollback()
                flash(
                    "UID already exists.",
                    "error"
                )

    cur.execute(
        """
        SELECT
            id,
            student_id,
            name
        FROM students
        WHERE id=%s
        """,
        (sid,)
    )

    student = cur.fetchone()

    cur.close()
    conn.close()

    if not student:
        return "Student not found", 404

    return render_template(
        "edit_student.html",
        student=student
    )


# ============================================================
# DELETE STUDENT
# ============================================================

@app.post(
    "/students/<int:sid>/delete"
)
@teacher_required
def delete_student(sid):

    conn = get_db()

    cur = conn.cursor()

    cur.execute(
        """
        DELETE FROM students
        WHERE id=%s
        """,
        (sid,)
    )

    conn.commit()

    cur.close()
    conn.close()

    flash(
        "Student deleted.",
        "success"
    )

    return redirect(
        url_for(
            "students"
        )
    )


# ============================================================
# ATTENDANCE RECORDS
# ============================================================

@app.get("/records")
@teacher_required
def records():

    selected = (
        request.args.get(
            "date"
        )
        or
        date.today().isoformat()
    )

    try:

        day = date.fromisoformat(
            selected
        )

    except ValueError:

        day = date.today()

    rows = attendance_for_day(
        day
    )

    return render_template(
        "records.html",

        rows=rows,

        selected=
            day.isoformat()
    )


# ============================================================
# CSV EXPORT
# ============================================================

@app.get("/export/attendance")
@teacher_required
def export_attendance():

    selected = request.args.get("date", "").strip()

    conn = get_db()
    cur = conn.cursor(dictionary=True)

    if selected:

        try:
            day = date.fromisoformat(selected)

        except ValueError:

            cur.close()
            conn.close()

            flash("Invalid date.", "error")

            return redirect(url_for("records"))

        cur.execute("""
            SELECT
                s.student_id,
                s.name,
                %s AS attendance_date,
                a.attendance_time,
                COALESCE(a.status, 'Absent') AS status

            FROM students s

            LEFT JOIN attendance a
                ON a.student_id = s.id
                AND a.attendance_date = %s

            ORDER BY s.name
        """, (day, day))

        rows = cur.fetchall()
        filename = f"attendance_{day.isoformat()}.csv"

    else:

        cur.execute("""
            SELECT
                s.student_id,
                s.name,
                a.attendance_date,
                a.attendance_time,
                a.status

            FROM attendance a

            INNER JOIN students s
                ON s.id = a.student_id

            ORDER BY
                a.attendance_date DESC,
                a.attendance_time DESC
        """)

        rows = cur.fetchall()
        filename = "attendance_all.csv"

    cur.close()
    conn.close()

    output = StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Student ID",
        "Name",
        "Attendance Date",
        "Attendance Time",
        "Status"
    ])

    for row in rows:
        writer.writerow([
            row["student_id"],
            row["name"],
            row["attendance_date"],
            row["attendance_time"] or "",
            row["status"]
        ])

    response = Response(
        output.getvalue(),
        mimetype="text/csv"
    )

    response.headers["Content-Disposition"] = (
        f"attachment; filename={filename}"
    )

    return response


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    try:

        conn = get_db()

        cur = conn.cursor()

        cur.execute(
            "SELECT DATABASE()"
        )

        database = (
            cur.fetchone()[0]
        )

        cur.close()
        conn.close()

        return jsonify(
            ok=True,
            database=database
        )

    except Exception as e:

        return jsonify(
            ok=False,
            error=str(e)
        ), 500


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    try:

        download_models()

        ensure_teacher()

    except Exception as e:

        print(
            "\nSTARTUP ERROR:",
            e
        )

        print(
            "\nMake sure:"
        )

        print(
            "1. MySQL is running."
        )

        print(
            "2. database.sql has been executed."
        )

        print(
            "3. .env contains the correct "
            "MySQL password."
        )

        print()

        raise

    print(
        "Student attendance:"
    )

    print(
        "http://127.0.0.1:5000/attendance"
    )

    print(
        "Teacher login:"
    )

    print(
        "http://127.0.0.1:5000/teacher/login"
    )

    print(
        "Analytics:"
    )

    print(
        "http://127.0.0.1:5000/analytics"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )