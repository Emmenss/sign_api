"""
API Flask — Reconnaissance Langue des Signes
Classes correctes tirées du notebook : hello, thanks, Iloveyou, no, yes
Optimisé pour minimiser la latence sur Render Free
"""

import base64
import logging
import os

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ⚠️ CLASSES EXACTES DU NOTEBOOK — NE PAS CHANGER L'ORDRE
# ─────────────────────────────────────────────
ACTIONS = np.array(["hello", "thanks", "Iloveyou", "no", "yes"])

SEQUENCE_LENGTH = 30   # identique au notebook
THRESHOLD       = 0.7  # seuil de confiance

app = Flask(__name__)

# ── MediaPipe ────────────────────────────────
mp_holistic = mp.solutions.holistic
_holistic   = None

def get_holistic():
    global _holistic
    if _holistic is None:
        # model_complexity=0 = plus rapide, suffisant pour keypoints
        _holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        logger.info("✅ MediaPipe Holistic initialisé (complexity=0)")
    return _holistic

# ── Modèle ───────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        path = os.path.join(os.path.dirname(__file__), "action.h5")
        if not os.path.exists(path):
            raise FileNotFoundError("action.h5 introuvable à la racine du projet")
        logger.info("⏳ Chargement du modèle...")
        _model = tf.keras.models.load_model(path)
        logger.info(f"✅ Modèle chargé — {_model.input_shape} → {_model.output_shape}")
    return _model

# ── Buffer par session ────────────────────────
_sequences: dict[str, list] = {}


# ─────────────────────────────────────────────
# EXTRACT KEYPOINTS — identique au notebook
# ordre : pose + face + lh + rh = 1662 valeurs
# ─────────────────────────────────────────────
def extract_keypoints(results) -> np.ndarray:
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )
    face = (
        np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(468 * 3)
    )
    lh = (
        np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])  # (1662,)


# ─────────────────────────────────────────────
# DÉCODAGE FRAME
# ─────────────────────────────────────────────
def decode_frame(b64: str, width: int, height: int):
    try:
        raw    = base64.b64decode(b64)
        np_arr = np.frombuffer(raw, np.uint8)

        # Essai JPEG (format le plus courant depuis Flutter)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            # Réduire la résolution pour accélérer MediaPipe
            frame = cv2.resize(frame, (320, 240))
            return frame

        # Fallback YUV — plan luminance seul
        if len(np_arr) >= width * height:
            gray  = np_arr[:width * height].reshape((height, width))
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame, (320, 240))
            return frame

        return None
    except Exception as e:
        logger.warning(f"decode_frame : {e}")
        return None


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Sign Language API",
        "status":  "online",
        "actions": ACTIONS.tolist(),
        "sequence_length": SEQUENCE_LENGTH,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "actions": ACTIONS.tolist(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Champ 'frame' manquant"}), 400

    width      = int(data.get("width",  640))
    height     = int(data.get("height", 480))
    session_id = data.get("session_id", "default")

    # 1. Décoder
    frame = decode_frame(data["frame"], width, height)
    if frame is None:
        return jsonify({"error": "Frame illisible"}), 400

    # 2. MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = get_holistic().process(rgb)

    # 3. Keypoints
    kp = extract_keypoints(results)

    # 4. Buffer glissant
    seq = _sequences.setdefault(session_id, [])
    seq.append(kp.tolist())
    if len(seq) > SEQUENCE_LENGTH:
        _sequences[session_id] = seq[-SEQUENCE_LENGTH:]

    buf_size = len(_sequences[session_id])

    # 5. Prédiction dès que le buffer est plein
    prediction    = None
    confidence    = 0.0
    probabilities = {}

    if buf_size == SEQUENCE_LENGTH:
        inp = np.expand_dims(_sequences[session_id], axis=0)  # [1, 30, 1662]
        res = get_model().predict(inp, verbose=0)[0]           # [5]

        probabilities = {ACTIONS[i]: float(round(res[i], 4)) for i in range(len(ACTIONS))}
        best          = int(np.argmax(res))
        confidence    = float(res[best])

        if confidence >= THRESHOLD:
            prediction = ACTIONS[best]

        # Décaler le buffer de 15 frames au lieu de tout garder
        # → nouvelle prédiction toutes les 15 frames (plus réactif)
        _sequences[session_id] = _sequences[session_id][15:]

    return jsonify({
        "prediction":    prediction,
        "confidence":    round(confidence, 4),
        "buffer_size":   buf_size,
        "probabilities": probabilities,
    })


@app.route("/reset", methods=["POST"])
def reset():
    data       = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id", "default")
    _sequences.pop(session_id, None)
    return jsonify({"status": "ok", "session_id": session_id})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Démarrage sur port {port}")
    logger.info(f"   Classes : {ACTIONS.tolist()}")
    app.run(host="0.0.0.0", port=port, debug=False)