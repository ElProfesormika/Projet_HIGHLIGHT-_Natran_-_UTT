# Guide pour pousser le projet sur GitHub

## Prérequis

1. **Installer Git** (si pas déjà installé) :
   - Télécharger depuis : https://git-scm.com/download/win
   - Ou installer via : `winget install Git.Git`

2. **Créer un compte GitHub** (si pas déjà fait) :
   - Aller sur : https://github.com

## Étapes pour pousser le projet

### 1. Initialiser le dépôt Git (si pas déjà fait)

```bash
git init
```

### 2. Ajouter le remote GitHub

```bash
git remote add origin https://github.com/ElProfesormika/Projet_HIGHLIGHT-_Natran_-_UTT.git
```

### 3. Vérifier le remote

```bash
git remote -v
```

### 4. Ajouter tous les fichiers

```bash
git add .
```

### 5. Faire le premier commit

```bash
git commit -m "Initial commit: Projet HIGHLIGHT+ - Détection de fuites de méthane avec IA"
```

### 6. Créer la branche main (si nécessaire)

```bash
git branch -M main
```

### 7. Pousser vers GitHub

```bash
git push -u origin main
```

## Commandes pour les mises à jour futures

Après avoir modifié des fichiers :

```bash
# Voir les fichiers modifiés
git status

# Ajouter les fichiers modifiés
git add .

# Faire un commit avec un message descriptif
git commit -m "Description des modifications"

# Pousser vers GitHub
git push
```

## Authentification GitHub

Si vous êtes demandé de vous authentifier :
- Utilisez un **Personal Access Token** (PAT) au lieu du mot de passe
- Créer un PAT : GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
- Scopes nécessaires : `repo` (accès complet aux dépôts)

## Structure du projet

Le projet contient :
- Code source Python (`highlight_plus/`)
- Interface Streamlit (`streamlit_app.py`)
- Documentation (`*.md`)
- Configuration (`CONFIG_OPTIMALE_CONCOURS.py`)
- Requirements (`requirements.txt`)

Le fichier `.gitignore` exclut automatiquement :
- Fichiers Python compilés (`__pycache__/`)
- Environnements virtuels (`venv/`)
- Fichiers temporaires
- Résultats de démonstration

