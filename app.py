"""
API Flask — Reconnaissance Langue des Signes
Déployée sur Render.com
"""

import base64
import os
import logging

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEQUENCE_LENGTH = 30
THRESHOLD       = 0.7

# ⚠️ Mets ici tes classes dans le même ordre qu'à l'entraînement
ACTIONS = np.array(['bonjour', 'merci', 'oui', 'non', 'boire'])

# ─────────────────────────────────────────────
# CHARGEMENT DU MODÈLE (lazy — au premier appel)
# Render a une RAM limitée, on évite de tout charger au démarrage
# ─────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        # Import ici pour ne pas bloquer le démarrage si TF est lent
        import tensorflow as tf
        model_path = os.path.join(os.path.dirname(__file__), 'action.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "action.h5 introuvable. Place le fichier à la racine du projet."
            )
        logger.info("⏳ Chargement du modèle...")
        _model = tf.keras.models.load_model(model_path)
        logger.info(f"✅ Modèle chargé — Input: {_model.input_shape}")
    return _model

# ─────────────────────────────────────────────
# MEDIAPIPE — instance unique réutilisée
# ─────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
_holistic    = None

def get_holistic():
    global _holistic
    if _holistic is None:
        _holistic = mp_holistic.Holistic(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            static_image_mode=False,
        )
    return _holistic

# ─────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────
app = Flask(__name__)

# Buffer de séquence par session — clé = identifiant client
# (simple dict en mémoire, suffisant pour une démo)
_sequences: dict[str, list] = {}


# ─────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────
def extract_keypoints(results) -> np.ndarray:
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility]
                  for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )
    face = (
        np.array([[r.x, r.y, r.z]
                  for r in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(468 * 3)
    )
    lh = (
        np.array([[r.x, r.y, r.z]
                  for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[r.x, r.y, r.z]
                  for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def decode_frame(b64_string: str, width: int, height: int):
    try:
        raw    = base64.b64decode(b64_string)
        np_arr = np.frombuffer(raw, np.uint8)

        # Essai JPEG
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            return frame

        # Fallback YUV plan Y → niveaux de gris
        if len(np_arr) >= width * height:
            gray = np_arr[:width * height].reshape((height, width))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

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
        "service": "Sign Language Recognition API",
        "status":  "online",
        "actions": ACTIONS.tolist(),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "actions": ACTIONS.tolist(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Body JSON :
    {
        "frame"     : "<base64>",
        "width"     : 640,
        "height"    : 480,
        "session_id": "device-uuid"   ← optionnel, mais recommandé
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Champ 'frame' manquant"}), 400

    width      = int(data.get("width",  640))
    height     = int(data.get("height", 480))
    session_id = data.get("session_id", "default")

    # Décoder la frame
    frame = decode_frame(data["frame"], width, height)
    if frame is None:
        return jsonify({"error": "Impossible de décoder la frame"}), 400

    # Détection MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = get_holistic().process(image_rgb)

    # Extraction keypoints
    keypoints = extract_keypoints(results)

    # Buffer par session
    seq = _sequences.setdefault(session_id, [])
    seq.append(keypoints.tolist())
    if len(seq) > SEQUENCE_LENGTH:
        _sequences[session_id] = seq[-SEQUENCE_LENGTH:]

    # Prédiction
    prediction    = None
    confidence    = 0.0
    probabilities = {}

    if len(_sequences[session_id]) == SEQUENCE_LENGTH:
        model      = get_model()
        input_data = np.expand_dims(_sequences[session_id], axis=0)
        res        = model.predict(input_data, verbose=0)[0]

        probabilities = {
            ACTIONS[i]: float(round(res[i], 4))
            for i in range(len(ACTIONS))
        }

        best_idx   = int(np.argmax(res))
        confidence = float(res[best_idx])

        if confidence >= THRESHOLD:
            prediction = ACTIONS[best_idx]

    return jsonify({
        "prediction":    prediction,
        "confidence":    round(confidence, 4),
        "buffer_size":   len(_sequences[session_id]),
        "probabilities": probabilities,
    })


@app.route("/reset", methods=["POST"])
def reset():
    data       = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id", "default")
    _sequences.pop(session_id, None)
    return jsonify({"status": "buffer réinitialisé", "session_id": session_id})


# ─────────────────────────────────────────────
# POINT D'ENTRÉE LOCAL (pas utilisé par Render)
# Render utilise : gunicorn app:app
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
