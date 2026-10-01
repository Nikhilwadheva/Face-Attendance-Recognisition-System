from __future__ import annotations

import csv
import hmac
import io
import os
import re
import secrets
from datetime import date
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import (Flask, jsonify, make_response, redirect, render_template,
                   request, session, url_for)

from database import init_app as init_database
from database import list_absent, list_attendance, list_subjects, record_attendance
from recognizer import FaceRecognitionService, ImageInputError

load_dotenv()
SUBJECT_RE = re.compile(r"[A-Za-z0-9 _.&-]{1,40}")


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True, template_folder=".")
    admin_password = os.environ.get("ADMIN_PASSWORD") or secrets.token_urlsafe(9)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        DATABASE=os.environ.get("DATABASE", str(Path(app.instance_path) / "attendance.sqlite3")),
        FACE_DIR=os.environ.get("FACE_DIR", str(Path(app.instance_path) / "faces")),
        FACE_DISTANCE_THRESHOLD=float(os.environ.get("FACE_DISTANCE_THRESHOLD", "75")),
        ADMIN_USER=os.environ.get("ADMIN_USER", "admin"),
        ADMIN_PASSWORD=admin_password,
        REQUIRE_LIVENESS=os.environ.get("REQUIRE_LIVENESS", "1") == "1",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        MAX_CONTENT_LENGTH=24 * 1024 * 1024,
    )
    if test_config:
        app.config.update(test_config)
    if not os.environ.get("ADMIN_PASSWORD") and not test_config:
        app.logger.warning("ADMIN_PASSWORD not set. Temporary admin password: %s", admin_password)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    Path(app.config["FACE_DIR"]).mkdir(parents=True, exist_ok=True)
    init_database(app)
    face_service = FaceRecognitionService(
        app.config["DATABASE"], app.config["FACE_DIR"], app.config["FACE_DISTANCE_THRESHOLD"]
    )
    app.extensions["face_service"] = face_service

    def admin_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not session.get("admin"):
                if request.path.startswith("/api/"):
                    return jsonify(error="Admin login required."), 401
                return redirect(url_for("login", next=request.path))
            return view(*args, **kwargs)
        return wrapped

    def valid_date(value: str) -> bool:
        try:
            date.fromisoformat(value)
            return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))
        except ValueError:
            return False

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            user_ok = hmac.compare_digest(request.form.get("username", ""), app.config["ADMIN_USER"])
            pass_ok = hmac.compare_digest(request.form.get("password", ""), app.config["ADMIN_PASSWORD"])
            if user_ok and pass_ok:
                session.clear()
                session["admin"] = True
                target = request.args.get("next", "")
                return redirect(target if target.startswith("/") and not target.startswith("//") else url_for("records_page"))
            error = "Wrong username or password."
        return render_template("login.html", error=error), (401 if error else 200)

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("index"))

    @app.get("/")
    def index():
        return render_template("checkin.html", subjects=list_subjects(), admin=bool(session.get("admin")))

    @app.get("/enroll")
    @admin_required
    def enroll_page():
        return render_template("enroll.html")

    @app.get("/records")
    @admin_required
    def records_page():
        return render_template("records.html", today=date.today().isoformat())

    @app.post("/api/enroll")
    @admin_required
    def enroll():
        payload = request.get_json(silent=True) or {}
        employee_code = str(payload.get("employee_code", "")).strip()
        name = str(payload.get("name", "")).strip()
        frames = payload.get("frames")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", employee_code):
            return jsonify(error="Use 1-32 letters, numbers, underscores, or hyphens for the ID."), 400
        if not 2 <= len(name) <= 100:
            return jsonify(error="Name must be between 2 and 100 characters."), 400
        if not isinstance(frames, list) or not 3 <= len(frames) <= 8:
            return jsonify(error="Capture between 3 and 8 face samples."), 400
        try:
            sample_count = face_service.enroll(employee_code, name, frames)
        except ImageInputError as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            app.logger.exception("Face enrollment failed")
            return jsonify(error=str(exc)), 500
        return jsonify(message=f"{name} enrolled with {sample_count} face samples.", sample_count=sample_count), 201

    @app.post("/api/recognize")
    def recognize():
        payload = request.get_json(silent=True) or {}
        subject = str(payload.get("subject") or "General").strip()
        if not SUBJECT_RE.fullmatch(subject):
            return jsonify(error="Subject may use letters, numbers, spaces and - _ . & (max 40)."), 400
        frames = payload.get("frames")
        if frames is None and payload.get("frame"):  # single-frame clients (no liveness)
            if app.config["REQUIRE_LIVENESS"]:
                return jsonify(error="Send 6-20 frames so a blink can be verified."), 400
            frames = [payload["frame"]]
        if not isinstance(frames, list) or not 1 <= len(frames) <= 20:
            return jsonify(error="Send 6-20 frames so a blink can be verified."), 400
        try:
            if app.config["REQUIRE_LIVENESS"] and not face_service.check_liveness(frames):
                return jsonify(recognized=False, liveness=False,
                               message="Liveness check failed. Look at the camera and blink naturally."), 200
            matches = face_service.recognize_all(frames[len(frames) // 2])
        except ImageInputError as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            app.logger.exception("Face recognition failed")
            return jsonify(error=str(exc)), 500
        if not matches:
            return jsonify(recognized=False, liveness=True,
                           message="No registered face matched. Enroll the person or try again."), 200
        people = [{"employee_code": code, "name": name, "distance": dist,
                   "checked_in": record_attendance(emp_id, subject)}
                  for emp_id, code, name, dist in matches]
        first = people[0]
        return jsonify(recognized=True, liveness=True, subject=subject, people=people,
                       checked_in=first["checked_in"], employee_code=first["employee_code"], name=first["name"],
                       message=f"{sum(p['checked_in'] for p in people)} marked present, "
                               f"{sum(not p['checked_in'] for p in people)} already recorded for {subject}.")

    @app.get("/api/attendance")
    @admin_required
    def attendance():
        selected_date = request.args.get("date", date.today().isoformat())
        subject = request.args.get("subject", "").strip() or None
        if not valid_date(selected_date):
            return jsonify(error="Date must be a valid YYYY-MM-DD date."), 400
        records = list_attendance(selected_date, subject)
        absent = list_absent(selected_date, subject) if subject else []
        return jsonify(date=selected_date, subject=subject, records=records, absent=absent,
                       subjects=list_subjects(), present_count=len(records), absent_count=len(absent))

    @app.get("/api/attendance/export.csv")
    @admin_required
    def export_attendance():
        selected_date = request.args.get("date", date.today().isoformat())
        subject = request.args.get("subject", "").strip() or None
        if not valid_date(selected_date):
            return jsonify(error="Enter a valid date using YYYY-MM-DD format."), 400
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["Employee ID", "Name", "Subject", "Attendance Date", "Check-in Time", "Status"])
        for r in list_attendance(selected_date, subject):
            writer.writerow([r["employee_code"], r["name"], r["subject"], r["attendance_date"], r["checked_in_at"], "Present"])
        if subject:
            for r in list_absent(selected_date, subject):
                writer.writerow([r["employee_code"], r["name"], subject, selected_date, "", "Absent"])
        response = make_response(output.getvalue())
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        response.headers["Content-Disposition"] = f'attachment; filename="attendance-{selected_date}.csv"'
        return response

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
