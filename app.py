import os
import tempfile
import json
from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from werkzeug.utils import secure_filename
from PIL import Image

UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/tmp/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

mp_pose = mp.solutions.pose


def load_image_safe(filepath):
    try:
        pil_img = Image.open(filepath).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def analyze_image(filepath):
    img = load_image_safe(filepath)
    if img is None:
        return None, "invalid_image_format"

    h, w = img.shape[:2]

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as pose:

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            return None, "no_landmarks"

        lm = res.pose_landmarks.landmark
        def xy(i): return (lm[i].x * w, lm[i].y * h)

        left_shoulder = xy(11)
        right_shoulder = xy(12)
        left_hip = xy(23)
        right_hip = xy(24)
        nose = xy(0)

        shoulder_w = euclidean(left_shoulder, right_shoulder)
        hip_w = euclidean(left_hip, right_hip)

        try:
            ankle_left = xy(27)
            ankle_right = xy(28)
            ankle_y = (ankle_left[1] + ankle_right[1]) / 2
            height_px = abs(ankle_y - nose[1])
        except:
            height_px = None

        shoulder_hip_ratio = shoulder_w / (hip_w + 1e-9)
        shoulder_height_ratio = shoulder_w / (height_px + 1e-9) if height_px else None

        if shoulder_hip_ratio > 1.10:
            shape = "inverted_triangle"
        elif shoulder_hip_ratio < 0.90:
            shape = "triangle"
        else:
            waist_proxy = hip_w * 0.85
            if waist_proxy < shoulder_w * 0.90 and waist_proxy < hip_w * 0.90:
                shape = "hourglass"
            else:
                shape = "rectangle"

        output = {
            "shape": shape,
            "measurements_px": {
                "shoulder_width_px": shoulder_w,
                "hip_width_px": hip_w,
                "estimated_height_px": height_px,
                "shoulder_hip_ratio": shoulder_hip_ratio,
                "shoulder_height_ratio": shoulder_height_ratio
            },
            "notes": "Landmark-based estimation. Better results with front-facing full-body photo."
        }

        return output, None

@app.route('/analyze_body', methods=['POST'])
def analyze_body():
    if "image" not in request.files:
        return jsonify({"error": "Missing file: use 'image' field"}), 400

    file = request.files["image"]

    if not file.mimetype.startswith("image/"):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename or "upload.jpg")
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=filename).name
    file.save(temp_path)

    result, err = analyze_image(temp_path)

    try: os.remove(temp_path)
    except: pass

    if err == "no_landmarks":
        return jsonify({"error": "No body landmarks detected"}), 422

    if err:
        return jsonify({"error": err}), 500

    return jsonify(result), 200

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
