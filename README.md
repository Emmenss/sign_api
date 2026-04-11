# 🤟 API de Reconnaissance de Langue des Signes

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00C7B7?logo=google&logoColor=white)](https://mediapipe.dev/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render&logoColor=white)](https://render.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📋 Table des Matières

- [Aperçu](#aperçu)
- [Actions Reconnaissables](#actions-reconnaissables)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Installation Locale](#installation-locale)
- [Configuration](#configuration)
- [Endpoints API](#endpoints-api)
- [Intégration Flutter](#intégration-flutter)
- [Déploiement sur Render](#déploiement-sur-render)
- [Optimisations](#optimisations)
- [Dépannage](#dépannage)
- [Structure du Projet](#structure-du-projet)
- [Licence](#licence)

## 🎯 Aperçu

API RESTful pour la reconnaissance en temps réel de la langue des signes à partir d'un flux vidéo. Elle utilise **MediaPipe** pour l'extraction des keypoints du corps, des mains et du visage, et **TensorFlow/Keras** (modèle LSTM) pour la classification des séquences de gestes sur 30 frames.

### ✨ Caractéristiques

- **Reconnaissance temps réel** - Traitement de 30 frames par prédiction
- **Latence minimisée** - Optimisée pour les environnements gratuits (Render)
- **Gestion multi-sessions** - Plusieurs utilisateurs simultanés
- **Buffer glissant** - Prédictions toutes les 15 frames
- **Seuil de confiance ajustable** - Filtrage des prédictions peu sûres

## 🎬 Actions Reconnaissables

| Action | Signification | Emoji |
|--------|---------------|-------|
| **hello** | Bonjour / Salutation | 👋 |
| **thanks** | Merci | 🙏 |
| **Iloveyou** | Je t'aime | ❤️ |
| **no** | Non | 🙅 |
| **yes** | Oui | 👍 |

## 🏗 Architecture du Système

### Pipeline de Traitement

1. **Capture** → Image base64 envoyée par le client
2. **Décodage** → Conversion en format OpenCV (JPEG ou YUV)
3. **Redimensionnement** → 320x240 pixels pour accélérer MediaPipe
4. **Extraction** → 1662 keypoints (pose + visage + mains gauche/droite)
5. **Buffer** → Accumulation de 30 séquences
6. **Prédiction** → Modèle LSTM → Classe + Confidence
7. **Décalage** → Buffer réduit de 15 frames pour la prochaine prédiction

## 🛠 Technologies Utilisées

| Technologie | Version | Rôle |
|-------------|---------|------|
| **Python** | 3.9+ | Langage principal |
| **Flask** | 2.0+ | Framework API |
| **TensorFlow** | 2.x | Modèle de deep learning |
| **MediaPipe** | 0.10+ | Extraction keypoints |
| **OpenCV** | 4.x | Traitement image |
| **NumPy** | 1.x | Calculs matriciels |
| **Gunicorn** | 21.x | Serveur production |

### Dépendances Complètes (requirements.txt)

Flask==2.3.3
tensorflow==2.13.0
mediapipe==0.10.7
opencv-python==4.8.1.78
numpy==1.24.3
gunicorn==21.2.0

### 📦 Installation Locale
Prérequis

    Python 3.9 ou supérieur

    pip (gestionnaire de paquets)

    Git
### Étapes d'Installation
# 1. Cloner le dépôt
git clone https://github.com/votre-username/sign-language-api.git
cd sign-language-api

# 2. Créer un environnement virtuel
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Ajouter le modèle entraîné
# Placez votre fichier action.h5 à la racine
# (entraîné avec le notebook fourni)

# 5. Lancer l'API
python app.py

# 6. Tester l'API
curl http://localhost:5000/health

🔌 Endpoints API
1. Vérification du Service
http

GET /

Réponse :
json

{
  "service": "Sign Language API",
  "status": "online",
  "actions": ["hello", "thanks", "Iloveyou", "no", "yes"],
  "sequence_length": 30
}

2. Health Check
http

GET /health

Réponse :
json

{
  "status": "ok",
  "actions": ["hello", "thanks", "Iloveyou", "no", "yes"]
}

3. Prédiction (Endpoint Principal)
http

POST /predict
Content-Type: application/json

Body Requis :
json

{
  "frame": "base64_encoded_image_string",
  "width": 640,
  "height": 480,
  "session_id": "user_123"
}

Réponse avec prédiction (confiance ≥ seuil) :
json

{
  "prediction": "hello",
  "confidence": 0.8923,
  "buffer_size": 30,
  "probabilities": {
    "hello": 0.8923,
    "thanks": 0.0432,
    "Iloveyou": 0.0214,
    "no": 0.0198,
    "yes": 0.0233
  }
}

Réponse sans prédiction (confiance < seuil) :
json

{
  "prediction": null,
  "confidence": 0.6542,
  "buffer_size": 30,
  "probabilities": {
    "hello": 0.6542,
    "thanks": 0.1123,
    "Iloveyou": 0.0891,
    "no": 0.0722,
    "yes": 0.0722
  }
}

4. Réinitialisation de Session
http

POST /reset
Content-Type: application/json

Body :
json

{
  "session_id": "user_123"
}

Réponse :
json

{
  "status": "ok",
  "session_id": "user_123"
}
