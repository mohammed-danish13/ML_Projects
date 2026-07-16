# Gate — CNN Face Attendance System

A Flask + CNN attendance system: register faces via webcam, train a classifier,
then check in at a live "gate" that logs each recognized face against a
configurable deadline and flags latecomers.

## How it works

- **Register** (`/register`) — enter a name, capture ~30 webcam frames per
  user. Each frame is cropped to the face (OpenCV Haar cascade), converted to
  a 100x100 grayscale image, and saved under `dataset/<label_id>/`.
- **Train** — a small CNN (3 conv blocks + dense head, `models/cnn_model.py`)
  is trained from scratch over every registered user's samples, with light
  augmentation (rotation/shift/zoom/flip) since per-user sample counts are
  small. The trained model is saved to `trained_model/face_cnn.keras`.
- **Check In** (`/login`) — the browser scans continuously, sends frames to
  `/api/attendance/recognize`. If the CNN's confidence clears the configured
  threshold, attendance is logged once per user per day, timestamped, and
  marked `ON-TIME` or `LATE` by comparing against Settings.
- **Settings** (`/settings`) — set the daily login deadline, a grace period
  in minutes, and the recognition confidence threshold.
- **Dashboard** (`/dashboard`) — full check-in log for any date, with
  on-time / late / absent counts.

## Setup

```bash
cd face_attendance_system
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`. Your browser will ask for camera permission on
the Register and Check In pages — allow it (works over `localhost`; for LAN
access from another device you'll need HTTPS or a browser flag, since
`getUserMedia` requires a secure context).

## Typical flow

1. Go to **Register**, add each person (name + optional roll no.), run
   **Start Capture** for each (aim for 30+ samples, vary head angle/lighting).
2. Once at least 2 users are registered with 15+ samples each, click
   **Train Recognition Model**.
3. Set your deadline and grace period on **Settings**.
4. Point the **Check In** page at a webcam near the entrance — it logs
   check-ins automatically as people look at the camera.

## Notes / things to tune

- `IMG_SIZE` (100x100 grayscale) and the CNN architecture live in
  `models/cnn_model.py` — bump conv filters or add a block if you have a
  large roster and need more capacity.
- `recognition_threshold` (default 0.75) trades off false-accepts vs.
  false-rejects — lower it if legitimate users keep getting "not recognized",
  raise it if strangers get matched.
- Retraining is full-dataset-from-scratch, which is simplest and fine for
  classroom-sized rosters (tens of users). For hundreds of users you'd want
  incremental fine-tuning or an embedding-based approach (FaceNet-style)
  instead of a plain softmax classifier.
- SQLite DB lives at `database/attendance.db`, created automatically on
  first run.
