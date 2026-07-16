import os
from datetime import datetime, date, timedelta
from flask import Flask, render_template, request, jsonify

from database import db
from utils.face_utils import extract_face_from_dataurl, IMG_SIZE
from models.cnn_model import train_model, FaceRecognizer, DATASET_DIR

import cv2

app = Flask(__name__)
db.init_db()


# ---------------------------------------------------------------- pages ----

@app.route("/")
def index():
    summary = db.get_attendance_summary()
    return render_template("index.html", summary=summary)


@app.route("/register")
def register_page():
    return render_template("register.html", users=db.list_users())


@app.route("/login")
def login_page():
    deadline = db.get_setting("login_deadline")
    return render_template("login.html", deadline=deadline)


@app.route("/dashboard")
def dashboard_page():
    target_date = request.args.get("date") or date.today().isoformat()
    summary = db.get_attendance_summary(target_date)
    return render_template("dashboard.html", summary=summary)


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    if request.method == "POST":
        db.set_setting("login_deadline", request.form.get("login_deadline", "09:30"))
        db.set_setting("grace_minutes", request.form.get("grace_minutes", "5"))
        db.set_setting("recognition_threshold", request.form.get("recognition_threshold", "0.75"))
    settings = db.get_all_settings()
    return render_template("settings.html", settings=settings, users=db.list_users())


# ------------------------------------------------------------- API: users --

@app.route("/api/users", methods=["POST"])
def api_create_user():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    roll_no = (data.get("roll_no") or "").strip()
    if not name:
        return jsonify({"success": False, "error": "Name is required."}), 400

    user_id, label_id = db.create_user(name, roll_no)
    os.makedirs(os.path.join(DATASET_DIR, str(label_id)), exist_ok=True)
    return jsonify({"success": True, "user_id": user_id, "label_id": label_id})


@app.route("/api/users/<int:user_id>", methods=["DELETE"])
def api_delete_user(user_id):
    user = db.get_user_by_id(user_id)
    if not user:
        return jsonify({"success": False, "error": "User not found."}), 404
    db.delete_user(user_id)
    return jsonify({"success": True})


# ---------------------------------------------------------- API: capture --

@app.route("/api/register/capture", methods=["POST"])
def api_capture():
    data = request.get_json()
    user_id = data.get("user_id")
    frame = data.get("frame")
    user = db.get_user_by_id(user_id)
    if not user:
        return jsonify({"success": False, "error": "User not found."}), 404

    face, box = extract_face_from_dataurl(frame)
    if face is None:
        return jsonify({"success": False, "error": "No face detected. Center your face in the frame."})

    label_dir = os.path.join(DATASET_DIR, str(user["label_id"]))
    os.makedirs(label_dir, exist_ok=True)
    existing = len([f for f in os.listdir(label_dir) if f.endswith(".png")])
    out_path = os.path.join(label_dir, f"{existing + 1:03d}.png")
    cv2.imwrite(out_path, face)

    db.update_sample_count(user_id, 1)
    total = existing + 1
    return jsonify({"success": True, "sample_count": total, "box": box.tolist() if box is not None else None})


@app.route("/api/register/train", methods=["POST"])
def api_train():
    result = train_model()
    if result.get("success"):
        db.mark_all_trained()
        FaceRecognizer.reload()
    return jsonify(result)


# ------------------------------------------------------ API: recognition --

@app.route("/api/attendance/recognize", methods=["POST"])
def api_recognize():
    data = request.get_json()
    frame = data.get("frame")

    face, box = extract_face_from_dataurl(frame)
    if face is None:
        return jsonify({"success": False, "error": "no_face"})

    try:
        label_id, confidence = FaceRecognizer.predict(face)
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": "not_trained", "message": str(e)})

    threshold = float(db.get_setting("recognition_threshold", "0.75"))
    if confidence < threshold:
        return jsonify({"success": False, "error": "not_recognized", "confidence": round(confidence, 3)})

    user = db.get_user_by_label(label_id)
    if not user:
        return jsonify({"success": False, "error": "not_recognized", "confidence": round(confidence, 3)})

    status = _compute_status()
    result = db.mark_attendance(user["id"], status, confidence)

    return jsonify({
        "success": True,
        "name": user["name"],
        "roll_no": user["roll_no"],
        "confidence": round(confidence, 3),
        "status": result["record"]["status"],
        "time": result["record"]["time"],
        "already_marked": result["already_marked"],
    })


def _compute_status():
    """Compares now vs configured deadline + grace period."""
    deadline_str = db.get_setting("login_deadline", "09:30")
    grace = int(db.get_setting("grace_minutes", "5"))
    now = datetime.now()
    hh, mm = [int(p) for p in deadline_str.split(":")]
    deadline_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0) + timedelta(minutes=grace)
    return "LATE" if now > deadline_dt else "ON-TIME"


# --------------------------------------------------------- API: dashboard --

@app.route("/api/attendance")
def api_attendance():
    target_date = request.args.get("date") or date.today().isoformat()
    return jsonify(db.get_attendance_summary(target_date))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
