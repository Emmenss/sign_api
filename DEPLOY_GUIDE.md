# Déploiement sur Render — Guide complet
# ========================================

## Structure du dépôt GitHub à créer

    mon-api-signes/
    ├── app.py               ← l'API Flask
    ├── requirements.txt     ← dépendances Python
    ├── render.yaml          ← config Render (optionnel)
    └── action.h5            ← ⚠️ ton modèle (voir note ci-dessous)


## ⚠️ Problème avec action.h5

Le fichier .h5 est souvent trop lourd pour GitHub (limite 100 MB).
Deux solutions selon la taille de ton fichier :

### Solution A — Fichier < 100 MB : commit direct
    git add action.h5
    git commit -m "add model"
    git push

### Solution B — Fichier > 100 MB : Git LFS
    # Installer Git LFS
    git lfs install
    git lfs track "*.h5"
    git add .gitattributes action.h5
    git commit -m "add model with LFS"
    git push


## Étapes de déploiement sur Render

1. Va sur https://render.com et connecte-toi

2. Clique sur "New" → "Web Service"

3. Connecte ton dépôt GitHub

4. Configure le service :
   - Name        : sign-language-api (ou ce que tu veux)
   - Runtime     : Python 3
   - Build Command : pip install -r requirements.txt
   - Start Command : gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT
   - Plan        : Free (suffisant pour une démo)

5. Clique sur "Create Web Service"

6. Attends le build (5-10 minutes la première fois)
   → Render te donne une URL du type : https://sign-language-api.onrender.com


## Mettre à jour l'URL dans Flutter

Dans sign_recognition_service.dart, remplace :

    static const String _baseUrl = 'http://VOTRE_IP:5000';

par :

    static const String _baseUrl = 'https://sign-language-api.onrender.com';


## ⚠️ Important : cold start sur le plan gratuit

Sur le plan Free de Render, le serveur s'endort après 15 minutes d'inactivité.
Le premier appel après une pause prend ~30 secondes (cold start).

Pour gérer ça dans Flutter, augmente le timeout dans sign_recognition_service.dart :

    static const Duration _timeout = Duration(seconds: 30); // au lieu de 3

Et affiche un message à l'utilisateur lors du premier chargement.


## Tester l'API déployée

Ouvre dans ton navigateur :
    https://ton-service.onrender.com/health

Tu dois voir :
    {"actions": ["bonjour", "merci", ...], "status": "ok"}
