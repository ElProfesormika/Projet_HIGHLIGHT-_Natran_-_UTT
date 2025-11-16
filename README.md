# 🚁 HIGHLIGHT+ - Intelligent Methane Leak Detection System

> **Concours Innovation Natran x Fondation UTT**  
> *Optimisation des trajectoires de vol – Drones pour la détection de micro-fuites de méthane*

## 🎯 Vision du Projet

HIGHLIGHT+ est un système d'intelligence artificielle révolutionnaire qui transforme un drone-dirigeable en détective autonome de micro-fuites de méthane. Notre approche combine la rigueur de l'apprentissage actif avec l'efficacité énergétique d'un pilote automatique par apprentissage par renforcement profond.

### 🧠 Architecture Teacher-Student

- **Expert (Teacher)** : Planificateur stratégique basé sur les Processus Gaussiens
- **Apprenti (Student)** : Pilote tactique utilisant l'apprentissage par renforcement
- **Distillation de connaissance** : Transfert d'expertise du Teacher vers le Student
- **Objectif** : Maximiser le gain d'information tout en minimisant le coût énergétique

## 🚀 Installation Rapide

```bash
# Cloner le projet
git clone https://github.com/your-username/highlight-plus.git
cd highlight-plus

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'interface Streamlit (recommandé)
streamlit run streamlit_app.py
# OU
python launch_streamlit.py

# OU lancer la démonstration en ligne de commande
python demo.py
```

## 📊 Résultats Clés

D'après les tests expérimentaux et l'analyse détaillée :

| Métrique | Teacher (GP) | Student (RL) | Baseline Naïve | Amélioration |
|----------|--------------|-------------|----------------|--------------|
| **Taux de détection** | 85-92% | 92-95% | 12-15% | **+25% à +40%** |
| **Précision localisation** | 2.1m | 1.8m | >10m | **<2m erreur** |
| **Temps de détection** | 2-12s | 0.8-2.5s | 12.2s | **-93%** |
| **Efficacité énergétique** | 0.15 | 0.19 | 0.08 | **+26.7%** |
| **Score global** | 70-85/100 | 75-90/100 | 25-40/100 | **+100%** |

### Points Forts

- ✅ **Taux de succès mission** : 85-90% (détection dans tolérance de 10m)
- ✅ **Précision moyenne** : 1.8-2.1 mètres d'erreur
- ✅ **Intelligence adaptative** : Apprentissage en temps réel
- ✅ **Validation automatique** : Comparaison position réelle vs détectée

## 🏗️ Architecture du Système

```
highlight_plus/
├── models/           # Modèles IA (Teacher & Student)
│   ├── teacher_gp.py    # Expert - Processus Gaussiens
│   └── student_rl.py    # Apprenti - RL + Distillation
├── simulation/       # Simulateur physique et environnement
│   ├── environment.py   # Environnement Gymnasium
│   └── plume_model.py   # Modèle de panache de méthane
├── sensors/          # Simulateur de capteurs TDLAS
│   └── tdlas_sensor.py  # Capteur laser avec bruit réaliste
├── analysis/         # Analyse et validation
│   ├── enhanced_detector.py      # Détecteur multi-critères
│   ├── performance_validator.py # Validation de performance
│   └── learning_analysis.py      # Analyse de l'apprentissage
├── experiments/      # Scripts d'expérimentation
│   ├── run_comparison.py  # Comparaisons expérimentales
│   └── leak_position_test.py # Tests de robustesse
├── visualization/    # Outils de visualisation
│   └── plotter.py   # Graphiques et animations
└── utils/           # Utilitaires et helpers
    └── config_loader.py # Chargement de configuration
```

## 🧪 Utilisation

### Interface Streamlit (Recommandé)

```bash
streamlit run streamlit_app.py
# OU
python launch_streamlit.py
```

**Fonctionnalités de l'interface :**
- ✅ Configuration complète du système (panache, capteur, drone, IA)
- ✅ Gestion des positions de fuites multiples
- ✅ Simulation avec validation automatique
- ✅ Visualisation en temps réel
- ✅ Export des résultats (JSON, rapports)
- ✅ Métriques de performance conformes à l'analyse

### Démonstration en Ligne de Commande

```bash
# Démonstration complète
python demo.py

# Tests rapides
python test_quick.py
```

### Expérimentations

```bash
# Comparaison complète Teacher vs Student vs Baselines
python highlight_plus/experiments/run_comparison.py

# Tests de robustesse sur multiples positions
python highlight_plus/experiments/leak_position_test.py
```

## 📈 Métriques de Performance

Le système calcule un **score global (0-100)** basé sur :

```
Score_Global = 0.4 × Score_Détection + 0.4 × Score_Localisation + 0.2 × Score_Efficacité
```

**Interprétation des scores :**
- ✅ **80-100** : Excellent - Mission très réussie
- ℹ️ **60-79** : Bon - Mission réussie avec améliorations possibles
- ⚠️ **40-59** : Acceptable - Mission partielle
- ❌ **0-39** : Insuffisant - Mission échouée

**Scores moyens observés :**
- Teacher : 70-85/100
- Student : 75-90/100
- Baseline naïve : 25-40/100

## 🔬 Validation et Fiabilité

Le système inclut un **validateur de performance** automatique qui :
- Compare la position détectée avec la position réelle configurée
- Calcule l'erreur de localisation (distance et angle)
- Vérifie si la détection est dans la tolérance (10m par défaut)
- Génère des rapports détaillés avec toutes les métriques

**Pour prouver la fiabilité :**
1. Configurez une position de fuite dans l'interface (Onglet Configuration → Positions de Fuites)
2. Lancez la simulation (Onglet Simulation)
3. Le système détecte automatiquement cette position
4. Consultez les métriques de validation dans l'onglet "Résultats & Métriques"
5. Vérifiez la comparaison position réelle vs position détectée

## 📚 Documentation

- **[ANALYSE_APPRENTISSAGE_IA.md](ANALYSE_APPRENTISSAGE_IA.md)** - Analyse détaillée de l'IA et des performances (755 lignes)
- **[VALIDATION_PERFORMANCE.md](VALIDATION_PERFORMANCE.md)** - Documentation du système de validation
- **[VARIABLES_SYSTEME.md](VARIABLES_SYSTEME.md)** - Liste complète des variables configurables
- **[PRESENTATION_CONCOURS.md](PRESENTATION_CONCOURS.md)** - Présentation complète pour le concours
- **[LIVRABLES_CONCOURS.md](LIVRABLES_CONCOURS.md)** - Liste des livrables et guide d'utilisation
- **[ARCHITECTURE_DIAGRAMS.txt](ARCHITECTURE_DIAGRAMS.txt)** - Diagrammes d'architecture ASCII

## 🎓 Équipe

- **Housséni YABRE** - Lead AI Engineer
- **Kabinet SYLLA** - Simulation & Physics
- **Nobert Bassooma DIDANERA** - System Integration

## ⚠️ Note Importante

**Résultats en simulation** : Tous les résultats présentés sont obtenus en simulation avec validation automatique. Pour validation terrain, voir la feuille de route dans [PRESENTATION_CONCOURS.md](PRESENTATION_CONCOURS.md).

**Fiabilité** : Le système inclut une validation automatique qui compare les positions détectées avec les positions réelles configurées, permettant de prouver la fiabilité du modèle.

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE) pour plus de détails.

## 🤝 Contribution

Les contributions sont les bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour les guidelines.

---

*Développé avec ❤️ pour l'innovation environnementale*  
*Concours Innovation Natran x Fondation UTT - 2025*
