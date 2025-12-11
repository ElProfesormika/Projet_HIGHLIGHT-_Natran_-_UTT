# HIGHLIGHT+ - Intelligent Methane Leak Detection System

> **Concours Innovation Natran x Fondation UTT**  
> *Optimisation des trajectoires de vol – Drones pour la détection de micro-fuites de méthane*

## Vision du Projet

HIGHLIGHT+ est un système d'intelligence artificielle révolutionnaire qui transforme un drone-dirigeable en détective autonome de micro-fuites de méthane. Notre approche combine la rigueur de l'apprentissage actif avec l'efficacité énergétique d'un pilote automatique par apprentissage par renforcement profond.

### Architecture Teacher-Student

- **Expert (Teacher)** : Planificateur stratégique basé sur les Processus Gaussiens
- **Apprenti (Student)** : Pilote tactique utilisant l'apprentissage par renforcement
- **Distillation de connaissance** : Transfert d'expertise du Teacher vers le Student
- **Objectif** : Maximiser le gain d'information tout en minimisant le coût énergétique

## Installation Rapide

```bash
# Cloner le projet
git clone https://github.com/ElProfesormika/Projet_HIGHLIGHT-_Natran_-_UTT.git
cd Projet_HIGHLIGHT-_Natran_-_UTT

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'interface Streamlit (recommandé)
streamlit run streamlit_app.py
# OU
python launch_app.py

# OU lancer la démonstration en ligne de commande
python demo.py
```

## Résultats Clés

D'après les tests expérimentaux et l'analyse détaillée :

| Métrique | Teacher (GP) | Student (RL) | Baseline Naïve | Amélioration |
|----------|--------------|-------------|----------------|--------------|
| **Taux de détection** | 85-92% | 92-95% | 12-15% | **+25% à +40%** |
| **Précision localisation** | 2.1m | 1.8m | >10m | **<2m erreur** |
| **Temps de détection** | 2-12s | 0.8-2.5s | 12.2s | **-93%** |
| **Efficacité énergétique** | 0.15 | 0.19 | 0.08 | **+26.7%** |
| **Score global** | 70-85/100 | 75-90/100 | 25-40/100 | **+100%** |

### Points Forts

- **Taux de succès mission** : 85-90% (détection dans tolérance de 10m)
- **Précision moyenne** : 1.8-2.1 mètres d'erreur
- **Intelligence adaptative** : Apprentissage en temps réel avec stratégie adaptative Teacher-Student
- **Validation automatique** : Comparaison position réelle vs détectée
- **Validateur GP** : Estimation probabiliste de la position de fuite avec Processus Gaussiens
- **Détection multi-fuites** : Détection de toutes les positions de fuite avec probabilité élevée sur la carte GP
- **Extraction complète** : Retourne toutes les positions détectées (pas seulement la meilleure)
- **Tri intelligent** : Positions triées par probabilité GP décroissante (meilleure en premier)
- **Reconnaissance de zone** : Stratégie multi-phase pour convergence précise sans dépassement
- **Stratégie adaptative** : Teacher et Student s'ajustent dynamiquement selon la confiance du Student
- **Comparaison simplifiée** : Métriques dynamiques avec visualisations régénérées à chaque exécution
- **Métriques enrichies** : Précision de localisation, confiance moyenne, temps de détection

## Architecture du Système

Le projet est organisé en modules Python modulaires :

- **`highlight_plus/models/`** : Modèles IA (Teacher GP & Student RL)
- **`highlight_plus/simulation/`** : Simulateur physique et environnement Gymnasium
- **`highlight_plus/sensors/`** : Simulateur de capteurs TDLAS avec bruit réaliste
- **`highlight_plus/analysis/`** : Analyse, validation et détection multi-critères
- **`highlight_plus/experiments/`** : Scripts d'expérimentation et comparaisons
- **`highlight_plus/visualization/`** : Outils de visualisation et graphiques
- **`highlight_plus/utils/`** : Utilitaires et chargement de configuration
- **`highlight_plus/gui/`** : Interface graphique PyQt (alternative à Streamlit)
- **`highlight_plus/data/`** : Chargeur de données réelles pour validation

## Utilisation

### Interface Streamlit (Recommandé)

```bash
# Méthode 1 : Directement avec Streamlit
streamlit run streamlit_app.py

# Méthode 2 : Via le lanceur Python
python launch_app.py

# Méthode 3 : Sur Windows, utiliser le script batch
restart_streamlit.bat
```

**Fonctionnalités de l'interface :**
- Configuration complète du système (panache, capteur, drone, IA)
- Gestion des positions de fuites multiples
- Simulation avec validation automatique
- **Visualisation en temps réel** : Carte de confiance GP et trajectoire du drone
- **Détection multi-fuites** : Détection de toutes les positions avec probabilité élevée sur la carte GP
- **Affichage complet** : Toutes les positions détectées affichées avec leur probabilité GP
- **Tri automatique** : Positions triées par probabilité décroissante (meilleure en premier)
- **Statistiques précises** : La meilleure position (probabilité la plus élevée) est utilisée pour les métriques
- **Comparaison simplifiée** : Comparaison Naïve vs HIGHLIGHT+ avec métriques dynamiques
- **Figures dynamiques** : Graphiques et trajectoires régénérés à chaque exécution
- **Métriques enrichies** : Précision de localisation, confiance moyenne, temps de détection, score de précision
- Export des résultats (JSON, rapports)
- Métriques de performance conformes à l'analyse

### Démonstration en Ligne de Commande

```bash
# Démonstration complète avec résultats sauvegardés
python demo.py
```

Les résultats sont sauvegardés dans `demo_results/comparison_results.json`.

### Expérimentations

```bash
# Comparaison complète Teacher vs Student vs Baselines
python -m highlight_plus.experiments.run_comparison

# Tests de robustesse sur multiples positions
python -m highlight_plus.experiments.leak_position_test

# Test comparatif complet
python -m highlight_plus.experiments.comparative_test
```

### Configuration

Le fichier de configuration principal se trouve dans `configs/default.yaml`. Il permet de configurer :
- Paramètres du panache de méthane
- Caractéristiques du capteur TDLAS
- Paramètres du drone (vitesse, altitude, consommation énergétique)
- Hyperparamètres des modèles IA (Teacher GP et Student RL)

## Métriques de Performance

Le système calcule un **score global (0-100)** basé sur :

```
Score_Global = 0.4 × Score_Détection + 0.4 × Score_Localisation + 0.2 × Score_Efficacité
```

**Interprétation des scores :**
- **80-100** : Excellent - Mission très réussie
- **60-79** : Bon - Mission réussie avec améliorations possibles
- **40-59** : Acceptable - Mission partielle
- **0-39** : Insuffisant - Mission échouée

**Scores moyens observés :**
- Teacher : 70-85/100
- Student : 75-90/100
- Baseline naïve : 25-40/100

## Validation et Fiabilité

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

## Documentation

- **[DOCUMENTATION_FONCTIONS_VARIABLES.md](DOCUMENTATION_FONCTIONS_VARIABLES.md)** - Documentation complète de toutes les fonctions et variables du projet
- **[ANALYSE_APPRENTISSAGE_IA.md](ANALYSE_APPRENTISSAGE_IA.md)** - Analyse détaillée de l'IA et des performances
- **[MODELE_TRAJECTOIRE_DRONE.md](MODELE_TRAJECTOIRE_DRONE.md)** - Explication du modèle de trajectoire du drone
- **[DIFFERENCES_MODES.md](DIFFERENCES_MODES.md)** - Explication des différences entre les modes de simulation

## Équipe

- **Housséni YABRE** - ETUDIANT en Informatique et Systèmes d'Information à l'UTT
- **Kabinet SYLLA** - ETUDIANT en Informatique et Systèmes d'Information à l'UTT
- **Nobert Bassooma DIDANERA** - Etudiant en fin de Master IA et Big Data, En mobilité à l'UTT

## Contact

Pour toute question ou information complémentaire :
- **Email** : housseni.yabre@utt.fr , kabinet.sylla@utt.fr , bassooma_norbert.didanera@utt.fr

## Structure du Projet

```
Natran_x_UTT/
├── highlight_plus/          # Package principal
│   ├── models/              # Modèles IA (Teacher GP & Student RL)
│   ├── simulation/         # Simulateur physique et environnement
│   ├── sensors/            # Simulateur de capteurs TDLAS
│   ├── analysis/           # Analyse et validation
│   ├── experiments/        # Scripts d'expérimentation
│   ├── visualization/      # Outils de visualisation
│   ├── utils/              # Utilitaires et helpers
│   ├── gui/                # Interface graphique (PyQt)
│   └── data/               # Chargeur de données réelles
├── configs/                # Fichiers de configuration
│   └── default.yaml        # Configuration par défaut
├── demo_results/            # Résultats des démonstrations
├── streamlit_app.py        # Interface Streamlit principale
├── launch_app.py           # Lanceur de l'application
├── demo.py                 # Script de démonstration
├── restart_streamlit.bat   # Script de redémarrage (Windows)
├── requirements.txt        # Dépendances Python
├── LICENSE                 # Licence MIT
├── logo_UTT.png           # Logo UTT
├── logo_natran.png        # Logo NATRAN
└── README.md              # Ce fichier
```

## Note Importante

**Résultats en simulation** : Tous les résultats présentés sont obtenus en simulation avec validation automatique.

**Fiabilité** : Le système inclut une validation automatique qui compare les positions détectées avec les positions réelles configurées, permettant de prouver la fiabilité du modèle.

**Fonctionnalités avancées** :
- **Validateur GP** : Utilise un Processus Gaussien pour estimer la position de fuite avec probabilité
- **Détection multi-fuites** : Extrait toutes les positions avec probabilité élevée de la carte de confiance GP
- **Extraction complète** : Retourne toutes les positions détectées, pas seulement la meilleure
- **Tri intelligent** : Positions automatiquement triées par probabilité GP décroissante
- **Statistiques optimisées** : La meilleure position (probabilité la plus élevée) est utilisée pour toutes les métriques
- **Visualisation améliorée** : Carte de confiance GP en temps réel avec toutes les positions détectées clairement visibles
- **Mode Teacher-Student amélioré** : Intégration complète GP + Teacher + Student pour détection optimale
- **Stratégie adaptative** : Teacher et Student s'ajustent dynamiquement selon la confiance du Student (poids adaptatifs)
- **Comparaison simplifiée** : Section dédiée avec métriques dynamiques calculées à partir des résultats réels
- **Figures dynamiques** : Graphiques et trajectoires régénérés à chaque exécution pour refléter les dernières données
- **Métriques enrichies** : Précision de localisation, confiance moyenne, temps de détection, score de précision (0-100)

## Licence

MIT License - Voir [LICENSE](LICENSE) pour plus de détails.

---

*Développé pour l'innovation environnementale*  
*Concours Innovation Natran x Fondation UTT - 2025*
