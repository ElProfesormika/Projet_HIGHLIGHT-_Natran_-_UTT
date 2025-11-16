# Guide de Contribution - HIGHLIGHT+

Merci de votre intérêt pour contribuer au projet HIGHLIGHT+ ! Ce guide vous aidera à comprendre comment participer au développement.

## 🎯 Objectifs du Projet

HIGHLIGHT+ est un système d'intelligence artificielle pour la détection intelligente de micro-fuites de méthane utilisant une architecture Teacher-Student. Le projet vise à optimiser les trajectoires de vol de drones pour maximiser la détection tout en minimisant la consommation énergétique.

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.8+
- pip ou conda
- Git

### Installation
```bash
# Cloner le repository
git clone https://github.com/your-username/highlight-plus.git
cd highlight-plus

# Installer les dépendances
pip install -r requirements.txt

# Test rapide
python test_quick.py
```

## 📁 Structure du Projet

```
highlight_plus/
├── models/           # Modèles IA (Teacher & Student)
├── simulation/       # Simulateur physique et environnement
├── sensors/          # Simulateur de capteurs TDLAS
├── visualization/    # Outils de visualisation
├── experiments/      # Scripts d'expérimentation
└── utils/           # Utilitaires et helpers
```

## 🔧 Types de Contributions

### 🐛 Signalement de Bugs
1. Vérifiez que le bug n'a pas déjà été signalé
2. Créez une issue avec :
   - Description claire du problème
   - Étapes pour reproduire
   - Environnement (OS, Python version)
   - Logs d'erreur si disponibles

### ✨ Nouvelles Fonctionnalités
1. Créez une issue pour discuter de la fonctionnalité
2. Attendez l'approbation avant de commencer le développement
3. Suivez les conventions de code
4. Ajoutez des tests
5. Mettez à jour la documentation

### 📚 Amélioration de la Documentation
- Correction d'erreurs
- Ajout d'exemples
- Amélioration de la clarté
- Traduction

### 🧪 Tests et Qualité
- Ajout de tests unitaires
- Amélioration de la couverture de code
- Optimisation des performances
- Refactoring

## 📝 Conventions de Code

### Style de Code
- Utilisez `black` pour le formatage automatique
- Suivez PEP 8
- Utilisez des docstrings pour toutes les fonctions publiques
- Nommez les variables de manière descriptive

### Structure des Commits
```
type(scope): description courte

Description plus détaillée si nécessaire

Fixes #issue_number
```

Types de commits :
- `feat`: nouvelle fonctionnalité
- `fix`: correction de bug
- `docs`: documentation
- `test`: tests
- `refactor`: refactoring
- `perf`: optimisation

### Branches
- `main`: branche principale stable
- `develop`: branche de développement
- `feature/nom-fonctionnalite`: nouvelles fonctionnalités
- `bugfix/nom-bug`: corrections de bugs

## 🧪 Tests

### Exécution des Tests
```bash
# Test rapide
python test_quick.py

# Test complet
python -m pytest tests/

# Test avec couverture
python -m pytest --cov=highlight_plus tests/
```

### Ajout de Tests
- Créez des tests unitaires pour chaque nouvelle fonctionnalité
- Utilisez des noms descriptifs pour les tests
- Testez les cas limites et les erreurs

## 📖 Documentation

### Documentation du Code
- Utilisez des docstrings au format Google
- Documentez tous les paramètres et valeurs de retour
- Ajoutez des exemples d'utilisation

### Documentation Utilisateur
- Mettez à jour le README.md si nécessaire
- Ajoutez des exemples dans la documentation
- Créez des tutoriels pour les nouvelles fonctionnalités

## 🔄 Processus de Contribution

1. **Fork** le repository
2. **Clone** votre fork localement
3. **Créez** une branche pour votre contribution
4. **Développez** votre fonctionnalité/correction
5. **Testez** votre code
6. **Commitez** avec un message descriptif
7. **Pushez** vers votre fork
8. **Créez** une Pull Request

### Pull Request
- Titre descriptif
- Description détaillée des changements
- Référence aux issues concernées
- Screenshots si applicable
- Checklist des tests

## 🏗️ Architecture Technique

### Modèles IA
- **Teacher**: Processus Gaussiens pour l'apprentissage actif
- **Student**: Apprentissage par renforcement avec distillation

### Simulation
- **Plume Model**: Modèle physique de diffusion du méthane
- **TDLAS Sensor**: Simulateur de capteur laser
- **Environment**: Environnement Gymnasium pour RL

### Visualisation
- **Plotter**: Outils de visualisation avec matplotlib/plotly
- **Metrics**: Calcul et affichage des métriques de performance

## 🎓 Ressources d'Apprentissage

### Concepts Clés
- Apprentissage par renforcement
- Processus Gaussiens
- Distillation de connaissance
- Détection de gaz
- Optimisation multi-objectifs

### Documentation Externe
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Scikit-learn GP](https://scikit-learn.org/stable/modules/gaussian_process.html)

## 🤝 Code de Conduite

### Nos Engagements
- Environnement accueillant et inclusif
- Respect mutuel
- Collaboration constructive
- Focus sur le bien commun

### Comportements Inacceptables
- Langage offensant ou discriminatoire
- Harcèlement
- Spam ou publicité
- Divulgation d'informations privées

## 📞 Contact

- **Email**: highlight.plus@utt.fr
- **Issues**: Utilisez GitHub Issues
- **Discussions**: GitHub Discussions

## 📄 Licence

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.

---

Merci de contribuer à HIGHLIGHT+ ! 🚁✨









