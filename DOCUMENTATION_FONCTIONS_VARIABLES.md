# Documentation Complète des Fonctions et Variables - HIGHLIGHT+

**Version** : 1.0.0  
**Date** : 2025  
**Système** : HIGHLIGHT+ - Détection Intelligente de Micro-Fuites de Méthane

---

## Table des Matières

1. [Introduction](#introduction)
2. [Architecture Globale](#architecture-globale)
3. [Fichier Principal : streamlit_app.py](#fichier-principal-streamlit_apppy)
   - [Classes](#classes)
   - [Fonctions Helper](#fonctions-helper)
   - [Fonction Principale](#fonction-principale)
   - [Fonctions d'Affichage des Onglets](#fonctions-daffichage-des-onglets)
   - [Fonctions de Simulation](#fonctions-de-simulation)
   - [Fonctions de Génération de Résultats](#fonctions-de-génération-de-résultats)
   - [Fonctions d'Affichage des Résultats](#fonctions-daffichage-des-résultats)
4. [Variables de Session State](#variables-de-session-state)
5. [Modules highlight_plus](#modules-highlight_plus)
   - [Simulation](#simulation)
   - [Capteurs](#capteurs)
   - [Modèles IA](#modèles-ia)
   - [Analyse](#analyse)
6. [Flux de Données et Exécution](#flux-de-données-et-exécution)
7. [Notes Importantes](#notes-importantes)

---

## Introduction

Ce document fournit une documentation complète et structurée de toutes les fonctions, classes et variables du système HIGHLIGHT+. Il sert de référence pour comprendre l'architecture, le fonctionnement et l'utilisation du système.

**Public cible** :
- Développeurs souhaitant contribuer au projet
- Utilisateurs souhaitant comprendre le fonctionnement interne
- Évaluateurs du concours

**Structure** :
- Documentation détaillée de chaque composant
- Exemples d'utilisation
- Description des flux de données
- Notes sur les points d'attention

---

## Architecture Globale

```
HIGHLIGHT+
│
├── Interface Utilisateur (Streamlit)
│   ├── Configuration
│   ├── Simulation
│   └── Comparaison
│
├── Simulation Physique
│   ├── Modèle de Panache (MethanePlume)
│   ├── Capteur TDLAS (TDLASSensor)
│   └── Environnement (MethaneDetectionEnv)
│
├── Intelligence Artificielle
│   ├── Teacher (GaussianProcessTeacher)
│   ├── Student (StudentRL)
│   └── Stratégie Adaptative
│
└── Analyse et Validation
    ├── Détecteur Amélioré (EnhancedDetector)
    └── Validateur GP (MethaneLeakValidator)
```

---

## Fichier Principal : streamlit_app.py

### Classes

#### `MultiSourcePlume`

**Localisation** : Lignes 29-131  
**Fichier** : `streamlit_app.py`

**Description** : Classe wrapper pour gérer plusieurs sources de fuite en combinant leurs concentrations. Utilisée pour la détection multi-fuites.

**Méthodes** :

##### `__init__(leak_positions: list, base_config: dict)`
Initialise le wrapper avec plusieurs positions de fuite.

**Paramètres** :
- `leak_positions` : Liste de tuples `(x, y, intensity)` - Positions et intensités des fuites
- `base_config` : Dictionnaire - Configuration de base (vent, diffusion, etc.)

**Effets** :
- Crée un panache `MethanePlume` pour chaque source
- Stocke la configuration de référence

**Variables d'instance** :
- `leak_positions` : Liste de tuples (x, y, intensity)
- `plumes` : Liste des objets `MethanePlume`
- `config` : Configuration de référence (premier panache)

##### `concentration(x, y, time: float = 0.0)`
Calcule la concentration totale (somme de toutes les sources).

**Paramètres** :
- `x` : Float ou array - Coordonnée X
- `y` : Float ou array - Coordonnée Y
- `time` : Float (défaut: 0.0) - Temps

**Retourne** :
- Scalaire ou array - Concentration totale (somme de toutes les sources)

**Gestion** :
- Gère les scalaires et les arrays numpy
- Assure la cohérence des types

##### `gradient(x, y, time: float = 0.0)`
Calcule le gradient total (somme vectorielle des gradients).

**Paramètres** :
- `x` : Float ou array - Coordonnée X
- `y` : Float ou array - Coordonnée Y
- `time` : Float (défaut: 0.0) - Temps

**Retourne** :
- Tuple `(grad_x, grad_y)` - Gradients totaux

**Note** : Le gradient total est la somme vectorielle des gradients de chaque panache.

---

### Fonctions Helper

#### `get_base64_image(image_path: str) -> str`

**Localisation** : Ligne 566  
**Fichier** : `streamlit_app.py`

**Description** : Convertit une image en base64 pour l'affichage dans HTML.

**Paramètres** :
- `image_path` : String - Chemin vers le fichier image

**Retourne** :
- String - Base64 de l'image, ou chaîne vide si fichier non trouvé

**Utilisation** : Pour afficher les logos UTT et NATRAN dans le header

**Exemple** :
```python
logo_base64 = get_base64_image("logo_UTT.png")
```

---

#### `get_mode_display_name(mode_value: str) -> str`

**Localisation** : Ligne 575  
**Fichier** : `streamlit_app.py`

**Description** : Convertit une valeur de mode interne en nom d'affichage.

**Paramètres** :
- `mode_value` : String - Mode interne ('simple', 'teacher_student', 'full_learning')

**Retourne** :
- String - Nom d'affichage ('Mode Simple', 'Mode Teacher', 'Mode Teacher-Student')

**Mapping** :
- `'simple'` → `'Mode Simple'`
- `'teacher_student'` → `'Mode Teacher'`
- `'full_learning'` → `'Mode Teacher-Student'`

**Utilisation** : Pour afficher les noms de modes de manière cohérente dans l'interface

---

#### `log_message(message: str)`

**Localisation** : Ligne 4738  
**Fichier** : `streamlit_app.py`

**Description** : Ajoute un message au journal de simulation avec timestamp.

**Paramètres** :
- `message` : String - Message à logger

**Effets** :
- Ajoute le message à `st.session_state.simulation_logs` avec timestamp au format `[HH:MM:SS]`

**Format** : `[HH:MM:SS] {message}`

---

### Fonction Principale

#### `main()`

**Localisation** : Ligne 615  
**Fichier** : `streamlit_app.py`

**Description** : Fonction principale de l'application Streamlit. Point d'entrée de l'application.

**Actions** :
1. Configure la page Streamlit (titre, icône, layout)
2. Affiche le header avec logos UTT/NATRAN et titre HIGHLIGHT+
3. Initialise les variables de session state si nécessaire
4. Crée les onglets principaux :
   - **"Simulation"** : Simulation principale
   - **"Configuration"** : Configuration du système
   - **"Comparaison Simplifiée"** : Comparaison Naïve vs HIGHLIGHT+
5. Appelle les fonctions d'affichage correspondantes selon l'onglet sélectionné

**Flux** :
```
main() → onglet sélectionné → fonction d'affichage correspondante
```

---

### Fonctions d'Affichage des Onglets

#### `show_simulation_tab()`

**Localisation** : Ligne 654  
**Fichier** : `streamlit_app.py`

**Description** : Affiche l'onglet de simulation principal.

**Fonctionnalités** :
1. **Métriques de Configuration Rapide** :
   - Affiche les paramètres principaux (mode, position de fuite, seuil)
   - Affiche le nombre de fuites configurées (si multi-fuites)

2. **Bouton "Démarrer Simulation"** :
   - Lance `run_simulation()` si cliqué
   - Affiche un indicateur de progression

3. **Résultats de Simulation** :
   - Affiche les métriques de performance via `display_performance_metrics()`
   - Affiche la visualisation de trajectoire via `display_trajectory_visualization()`
   - Affiche les logs de simulation en temps réel

4. **Métriques en Temps Réel** :
   - Position du drone
   - Concentration mesurée
   - Énergie consommée
   - Temps écoulé

**Variables utilisées** :
- `st.session_state.plume_config`
- `st.session_state.sensor_config`
- `st.session_state.drone_config`
- `st.session_state.ai_config`
- `st.session_state.simulation_results`
- `st.session_state.simulation_logs`
- `st.session_state.detected_leaks`

---

#### `show_configuration_tab()`

**Localisation** : Ligne 765  
**Fichier** : `streamlit_app.py`

**Description** : Affiche l'onglet de configuration du système.

**Sections** :
1. **Configuration du Panache** (`show_plume_config()`) :
   - Position de la source (leak_x, leak_y)
   - Intensité de la fuite (leak_intensity)
   - Vitesse et direction du vent
   - Coefficients de dispersion

2. **Configuration du Capteur** (`show_sensor_config()`) :
   - Seuil de détection
   - Niveau de bruit
   - Portées min/max

3. **Configuration du Drone** (`show_drone_config()`) :
   - Position initiale
   - Altitude initiale
   - Vitesse maximale

4. **Configuration IA** (`show_ai_config()`) :
   - Mode de simulation (Simple, Teacher, Teacher-Student)
   - Nombre d'étapes maximum

5. **Gestion des Positions de Fuites** (`show_leak_positions_config()`) :
   - Ajouter/supprimer des positions
   - Activer/désactiver des positions
   - Liste des positions configurées

**Stockage** : Toutes les configurations sont stockées dans `st.session_state`

---

#### `show_plume_config()`

**Localisation** : Ligne 786  
**Fichier** : `streamlit_app.py`

**Description** : Interface de configuration du panache de méthane.

**Paramètres configurables** :
- `leak_x`, `leak_y` : Position de la source (m)
- `leak_intensity` : Intensité de la fuite (kg/s)
- `wind_speed` : Vitesse du vent (m/s)
- `wind_direction` : Direction du vent (degrés)
- `sigma_x`, `sigma_y` : Coefficients de dispersion (m)

**Stockage** : `st.session_state.plume_config`

**Note** : Si des positions de fuite personnalisées sont configurées, cette section affiche un message informatif et désactive les champs.

---

#### `show_sensor_config()`

**Localisation** : Ligne 852  
**Fichier** : `streamlit_app.py`

**Description** : Interface de configuration du capteur TDLAS.

**Paramètres configurables** :
- `detection_threshold` : Seuil de détection (kg/m³)
- `noise_level` : Niveau de bruit (σ)
- `range_max` : Portée maximale (m)
- `range_min` : Portée minimale (m)

**Stockage** : `st.session_state.sensor_config`

---

#### `show_drone_config()`

**Localisation** : Ligne 886  
**Fichier** : `streamlit_app.py`

**Description** : Interface de configuration du drone.

**Paramètres configurables** :
- `initial_x`, `initial_y` : Position initiale (m)
- `initial_altitude` : Altitude initiale (m)
- `max_speed` : Vitesse maximale (m/s)

**Stockage** : `st.session_state.drone_config`

---

#### `show_ai_config()`

**Localisation** : Ligne 923  
**Fichier** : `streamlit_app.py`

**Description** : Interface de configuration de l'intelligence artificielle.

**Paramètres configurables** :
- `simulation_mode` : Mode de simulation
  - `'simple'` : Mode Simple (navigation basée sur gradient)
  - `'teacher_student'` : Mode Teacher (GP uniquement)
  - `'full_learning'` : Mode Teacher-Student (GP + RL avec stratégie adaptative)
- `max_steps` : Nombre d'étapes maximum

**Stockage** : `st.session_state.ai_config`

---

#### `show_leak_positions_config()`

**Localisation** : Ligne 4705  
**Fichier** : `streamlit_app.py`

**Description** : Interface de gestion des positions de fuites multiples.

**Fonctionnalités** :
1. **Ajouter une Nouvelle Position** :
   - Formulaire avec champs X, Y, Intensité
   - Bouton "Ajouter Position"

2. **Gestion des Positions Existantes** :
   - Tableau avec toutes les positions configurées
   - Boutons pour activer/désactiver chaque position
   - Bouton pour supprimer une position

3. **Affichage** :
   - Liste des positions actives
   - Nombre total de positions

**Stockage** : `st.session_state.leak_positions` (liste de dictionnaires)

**Format des positions** :
```python
{
    'x': float,        # Position X (m)
    'y': float,        # Position Y (m)
    'intensity': float, # Intensité (kg/s)
    'active': bool     # Si la position est active
}
```

---

#### `show_comparative_simple_tab()`

**Localisation** : Ligne 1094  
**Fichier** : `streamlit_app.py`

**Description** : Affiche l'onglet de comparaison simplifiée Naïve vs HIGHLIGHT+.

**Fonctionnalités** :
1. **Formulaire de Configuration** :
   - Position de la fuite
   - Position de départ
   - Nombre d'étapes
   - Nombre de runs
   - Paramètres du panache

2. **Génération de Résultats** :
   - Appelle `generate_comparative_results()` pour générer les métriques
   - Stocke les résultats dans `st.session_state.comparative_metrics`

3. **Affichage des Métriques** :
   - Appelle `display_comparative_results()` pour afficher les résultats
   - Métriques dynamiques calculées à chaque exécution

4. **Graphiques** :
   - Graphiques de performance comparatifs
   - Comparaison des trajectoires

**Fonctions appelées** :
- `generate_comparative_results()`
- `display_comparative_results()`
- `generate_comparative_charts()`
- `generate_trajectory_comparison()`

---

### Fonctions de Simulation

#### `run_simulation()`

**Localisation** : Ligne 2768  
**Fichier** : `streamlit_app.py`

**Description** : Exécute une simulation complète de détection de fuite.

**Paramètres** (via session_state) :
- Configuration du panache (`st.session_state.plume_config`)
- Configuration du capteur (`st.session_state.sensor_config`)
- Configuration du drone (`st.session_state.drone_config`)
- Configuration IA (`st.session_state.ai_config`)
- Positions de fuite (`st.session_state.leak_positions`)

**Processus** :

1. **Initialisation** :
   - Crée l'environnement de simulation (`MethaneDetectionEnv`)
   - Initialise le panache (simple ou multi-sources)
   - Initialise le capteur TDLAS
   - Initialise le Teacher (GP) et Student (RL) selon le mode
   - Initialise le détecteur amélioré et le validateur GP

2. **Boucle de Simulation** :
   Pour chaque étape :
   - **Observation** : Récupère l'état actuel
   - **Sélection d'Action** :
     - Mode Simple : Navigation basée sur gradient
     - Mode Teacher : Action du Teacher (GP)
     - Mode Teacher-Student : Combinaison adaptative Teacher + Student
   - **Exécution** : Exécute l'action dans l'environnement
   - **Mesure** : Mesure la concentration avec le capteur TDLAS
   - **Mise à jour Teacher** : Ajoute l'observation au GP
   - **Mise à jour Student** : Stocke l'expérience et apprend (si mode Teacher-Student)
   - **Validation** : Valide les détections via le détecteur amélioré
   - **Extraction GP** : Extrait toutes les positions de fuite depuis la carte GP
   - **Mise à jour Métriques** : Met à jour les métriques en temps réel
   - **Visualisation** : Met à jour la visualisation en temps réel (toutes les 5 étapes)

3. **Post-Traitement** :
   - Calcule les métriques finales
   - Identifie la meilleure position estimée (probabilité GP la plus élevée)
   - Stocke tous les résultats

**Retourne** : Dictionnaire `results` contenant :

```python
{
    'trajectory': list,              # Liste des positions du drone
    'detections': list,              # Liste des détections
    'total_energy': float,           # Énergie totale consommée (J)
    'n_detections': int,             # Nombre de détections
    'max_concentration': float,      # Concentration maximale mesurée
    'total_reward': float,           # Récompense totale
    'total_time': float,             # Temps total de simulation (s)
    'detection_rate': float,         # Taux de détection
    'energy_efficiency': float,      # Efficacité énergétique
    'performance_metrics': object,   # Objet PerformanceMetrics
    'performance_report': str,       # Rapport de performance
    'detector_stats': dict,          # Statistiques du détecteur
    'estimated_position': array,     # Position estimée (meilleure)
    'estimation_confidence': float,  # Confiance de l'estimation (0-1)
    'all_estimated_positions': list, # Toutes les positions détectées
    'auto_stopped': bool,            # Indicateur d'arrêt automatique
    'gp_validator_used': bool,      # Indicateur d'utilisation du validateur GP
    'use_multi_source': bool,        # Indicateur de mode multi-fuites
    'all_leak_positions': list       # Liste de toutes les positions de fuite réelles
}
```

**Variables de session_state modifiées** :
- `st.session_state.detected_leaks` : Liste de toutes les positions détectées avec probabilité GP (triée par probabilité décroissante)
- `st.session_state.simulation_results` : Résultats complets de la simulation
- `st.session_state.simulation_progress` : Progression de la simulation (0-100)
- `st.session_state.simulation_logs` : Logs de simulation avec timestamps

**Points d'attention** :
- **Multi-fuites** : Le système continue la recherche après chaque détection
- **Probabilité GP** : Seuil minimum de 75% pour valider une détection
- **Meilleure Position** : La position avec la probabilité GP la plus élevée est utilisée pour les statistiques
- **Extraction GP** : Toutes les positions avec probabilité > 75% sont extraites de la carte GP et stockées

---

### Fonctions de Génération de Résultats

#### `generate_comparative_results(leak_x, leak_y, start_x, start_y, max_steps, n_runs, threshold, intensity, wind_speed=2.0, wind_direction=45, sigma_x=5.0)`

**Localisation** : Ligne 1176  
**Fichier** : `streamlit_app.py`

**Description** : Génère des résultats comparatifs entre stratégie naïve (zigzag) et HIGHLIGHT+.

**Paramètres** :
- `leak_x`, `leak_y` : Float - Position de la fuite (m)
- `start_x`, `start_y` : Float - Position de départ du drone (m)
- `max_steps` : Int - Nombre maximum d'étapes
- `n_runs` : Int - Nombre de runs pour la moyenne
- `threshold` : Float - Seuil de détection (kg/m³)
- `intensity` : Float - Intensité de la fuite (kg/s)
- `wind_speed` : Float (défaut: 2.0) - Vitesse du vent (m/s)
- `wind_direction` : Float (défaut: 45) - Direction du vent (degrés)
- `sigma_x` : Float (défaut: 5.0) - Coefficient de dispersion X (m)

**Processus** :
1. **Stratégie Naïve** :
   - Implémentation directe d'un agent zigzag
   - Navigation systématique en zigzag
   - Pas d'apprentissage

2. **Stratégie HIGHLIGHT+** :
   - Utilise le modèle complet Teacher-Student
   - Initialise `GaussianProcessTeacher` et `StudentRL`
   - Utilise `EnhancedDetector` avec GP Validator
   - Apprentissage en temps réel

3. **Calcul des Métriques** :
   - Pour chaque run, calcule les métriques
   - Moyenne sur tous les runs

**Retourne** : Dictionnaire `metrics` contenant :

```python
{
    'naive': {
        'detections': int,
        'energy': float,
        'time': float,
        'distance': float,
        'estimated_position': array,
        'estimation_confidence': float,
        'localization_error': float,
        'avg_confidence': float
    },
    'highlight': {
        'detections': int,
        'energy': float,
        'time': float,
        'distance': float,
        'estimated_position': array,
        'estimation_confidence': float,
        'localization_error': float,
        'avg_confidence': float
    }
}
```

**Stockage** : `st.session_state.comparative_metrics`

---

#### `generate_comparative_charts(metrics, return_buffer=False)`

**Localisation** : Ligne 1660  
**Fichier** : `streamlit_app.py`

**Description** : Génère des graphiques comparatifs de performance.

**Paramètres** :
- `metrics` : Dictionnaire - Métriques (résultat de `generate_comparative_results`)
- `return_buffer` : Bool (défaut: False) - Si True, retourne un buffer d'image

**Retourne** :
- Buffer d'image matplotlib si `return_buffer=True`, sinon None

**Graphiques générés** :
1. Nombre de détections (barres)
2. Énergie consommée (barres)
3. Temps de détection (barres)
4. Distance finale (barres)

**Stockage** : `st.session_state.comparative_charts`

---

#### `generate_trajectory_comparison(trajectory_naive, trajectory_highlight, true_leak_pos, all_leak_positions=None, return_buffer=True)`

**Localisation** : Ligne 1760  
**Fichier** : `streamlit_app.py`

**Description** : Génère une visualisation comparative des trajectoires.

**Paramètres** :
- `trajectory_naive` : Liste de tuples - Trajectoire de la stratégie naïve
- `trajectory_highlight` : Liste de tuples - Trajectoire de HIGHLIGHT+
- `true_leak_pos` : Tuple - Position réelle de la fuite
- `all_leak_positions` : Liste (optionnel) - Liste de toutes les positions de fuite (multi-fuites)
- `return_buffer` : Bool (défaut: True) - Si True, retourne un buffer d'image

**Retourne** : Buffer d'image matplotlib

**Visualisation** :
- Trajectoire naïve (bleu, ligne continue)
- Trajectoire HIGHLIGHT+ (rouge, ligne continue)
- Position réelle de fuite (jaune, marqueur X)
- Positions de fuite multiples (si applicable, marqueurs différents)

**Stockage** : `st.session_state.comparative_trajectory`

---

#### `generate_performance_report(metrics, n_runs, save_path=None)`

**Localisation** : Ligne 1852  
**Fichier** : `streamlit_app.py`

**Description** : Génère un rapport textuel de performance.

**Paramètres** :
- `metrics` : Dictionnaire - Métriques
- `n_runs` : Int - Nombre de runs
- `save_path` : String (optionnel) - Chemin de sauvegarde

**Retourne** : String - Texte du rapport

**Contenu du rapport** :
- Résumé exécutif
- Métriques comparatives détaillées
- Analyse des performances
- Recommandations

**Note** : Si `save_path` est fourni, le rapport est également sauvegardé dans un fichier.

---

### Fonctions d'Affichage des Résultats

#### `display_comparative_results()`

**Localisation** : Ligne 1930  
**Fichier** : `streamlit_app.py`

**Description** : Affiche les résultats comparatifs dans l'interface Streamlit.

**Fonctionnalités** :
1. **Résultats Comparatifs** :
   - Taux de Détection HIGHLIGHT+
   - Économie d'Énergie
   - Amélioration Détections
   - Distance Finale
   - Temps de Détection
   - Précision Localisation
   - Confiance Moyenne

2. **Détails de Localisation (GP Validator)** :
   - Position estimée
   - Probabilité GP
   - Position réelle
   - Erreur de localisation (distance et angle)
   - Score de précision

3. **Toutes les Positions Détectées (Carte GP)** :
   - Tableau avec toutes les positions détectées
   - Triées par probabilité GP décroissante
   - Colonnes : ID, Coordonnées, Probabilité GP, Erreur, Statut

4. **Visualisations Comparatives** :
   - Graphiques de performance (via `generate_comparative_charts()`)
   - Comparaison des trajectoires (via `generate_trajectory_comparison()`)

**Variables utilisées** :
- `st.session_state.comparative_metrics`
- `st.session_state.comparative_charts`
- `st.session_state.comparative_trajectory`

**Note** : Les figures sont régénérées à chaque exécution avec un timestamp unique pour forcer la mise à jour dans Streamlit.

---

#### `display_performance_metrics(results)`

**Localisation** : Ligne 2193  
**Fichier** : `streamlit_app.py`

**Description** : Affiche les métriques de performance d'une simulation.

**Paramètres** :
- `results` : Dictionnaire - Résultats (retourné par `run_simulation()`)

**Sections affichées** :

1. **Indicateurs de Performance** :
   - Détections (nombre)
   - Énergie Consommée (J)
   - Durée de Mission (s)

2. **Validation de Performance** :
   - Score Global (0-100)
   - Temps de Détection (s)
   - Précision Localisation (m)
   - Taux de Succès Mission (%)

3. **Statistiques du Détecteur Amélioré** :
   - Confiance Moyenne (%)
   - Détections Validées (nombre/taux)

4. **Position Estimée (GP Validator)** :
   - Toutes les positions estimées avec probabilité GP
   - Erreur de localisation pour chaque position
   - Triées par probabilité décroissante

5. **Détails de Localisation** :
   - Position Réelle (pour validation/comparaison)
   - Position Détectée (meilleure, estimation indépendante)
   - Erreur (distance et angle)
   - Validateur GP actif avec nombre de mesures accumulées

**Note** : La position détectée affichée dans "Détails de Localisation" est la meilleure position (probabilité GP la plus élevée) parmi toutes les positions détectées.

---

#### `display_trajectory_visualization(results)`

**Localisation** : Ligne 2504  
**Fichier** : `streamlit_app.py`

**Description** : Visualise la trajectoire du drone avec validation de position.

**Paramètres** :
- `results` : Dictionnaire - Résultats

**Visualisation** :
- Trajectoire complète du drone (ligne bleue)
- Position de départ (marqueur vert)
- Position estimée de fuite (marqueur rouge, étoile)
- Position réelle de fuite (marqueur jaune, X)
- Erreur de localisation (ligne pointillée rouge)
- Toutes les positions détectées (si multi-fuites)

**Technologie** : Plotly pour interactivité

---

#### `visualize_plume()`

**Localisation** : Ligne 2718  
**Fichier** : `streamlit_app.py`

**Description** : Visualise le panache de méthane configuré.

**Fonctionnalités** :
- Carte de concentration 2D (heatmap)
- Contours de concentration
- Position de la source (marqueur)
- Positions de fuite multiples (si configurées)

**Technologie** : Plotly pour interactivité

---

## Variables de Session State

Les variables suivantes sont stockées dans `st.session_state` pour maintenir l'état de l'application entre les rechargements de page.

### Configuration

#### `st.session_state.plume_config`
**Type** : Dictionnaire  
**Description** : Configuration du panache de méthane

**Contenu** :
```python
{
    'leak_x': float,          # Position X de la fuite (m)
    'leak_y': float,          # Position Y de la fuite (m)
    'leak_intensity': float,  # Intensité de la fuite (kg/s)
    'wind_speed': float,      # Vitesse du vent (m/s)
    'wind_direction': float,  # Direction du vent (degrés)
    'sigma_x': float,         # Coefficient de dispersion X (m)
    'sigma_y': float          # Coefficient de dispersion Y (m)
}
```

**Initialisation** : Dans `show_plume_config()` si non existant

---

#### `st.session_state.sensor_config`
**Type** : Dictionnaire  
**Description** : Configuration du capteur TDLAS

**Contenu** :
```python
{
    'detection_threshold': float,  # Seuil de détection (kg/m³)
    'noise_level': float,          # Niveau de bruit (σ)
    'range_max': float,            # Portée maximale (m)
    'range_min': float             # Portée minimale (m)
}
```

**Initialisation** : Dans `show_sensor_config()` si non existant

---

#### `st.session_state.drone_config`
**Type** : Dictionnaire  
**Description** : Configuration du drone

**Contenu** :
```python
{
    'initial_x': float,            # Position X initiale (m)
    'initial_y': float,            # Position Y initiale (m)
    'initial_altitude': float,    # Altitude initiale (m)
    'max_speed': float             # Vitesse maximale (m/s)
}
```

**Initialisation** : Dans `show_drone_config()` si non existant

---

#### `st.session_state.ai_config`
**Type** : Dictionnaire  
**Description** : Configuration de l'intelligence artificielle

**Contenu** :
```python
{
    'simulation_mode': str,  # 'simple', 'teacher_student', 'full_learning'
    'max_steps': int         # Nombre maximum d'étapes
}
```

**Initialisation** : Dans `show_ai_config()` si non existant

---

#### `st.session_state.leak_positions`
**Type** : Liste de dictionnaires  
**Description** : Liste des positions de fuite (multi-fuites)

**Contenu** :
```python
[
    {
        'x': float,        # Position X (m)
        'y': float,        # Position Y (m)
        'intensity': float, # Intensité (kg/s)
        'active': bool     # Si la position est active
    },
    ...
]
```

**Initialisation** : Liste vide par défaut, remplie via `show_leak_positions_config()`

---

### État de Simulation

#### `st.session_state.simulation_running`
**Type** : Booléen  
**Description** : Indique si une simulation est en cours

**Valeurs** :
- `True` : Simulation en cours
- `False` : Aucune simulation en cours

---

#### `st.session_state.simulation_progress`
**Type** : Float (0-100)  
**Description** : Progression de la simulation en pourcentage

**Mise à jour** : Dans `run_simulation()` à chaque étape

---

#### `st.session_state.simulation_start_time`
**Type** : Float (timestamp)  
**Description** : Temps de début de la simulation

**Utilisation** : Pour calculer la durée de simulation

---

#### `st.session_state.simulation_results`
**Type** : Dictionnaire  
**Description** : Résultats complets de la dernière simulation

**Contenu** : Voir section `run_simulation()` pour la structure complète

**Mise à jour** : Par `run_simulation()` à la fin de la simulation

---

#### `st.session_state.detected_leaks`
**Type** : Liste de dictionnaires  
**Description** : Liste de toutes les positions de fuite détectées avec probabilité GP

**Contenu** :
```python
[
    {
        'position': [float, float],  # Position [x, y] (m)
        'confidence': float,          # Probabilité GP (0-1)
        'step': int,                  # Étape de détection
        'time': float                 # Temps de détection (s)
    },
    ...
]
```

**Note** : 
- Triée par probabilité GP décroissante (meilleure en premier)
- Seuil minimum de 75% pour être inclus
- Mise à jour dans `run_simulation()` lors de l'extraction GP

---

### Logs

#### `st.session_state.simulation_logs`
**Type** : Liste de strings  
**Description** : Journal des messages de simulation avec timestamps

**Format** : `[HH:MM:SS] {message}`

**Ajout** : Via `log_message()`

**Affichage** : Dans `show_simulation_tab()` dans un conteneur défilable

---

### Résultats Comparatifs

#### `st.session_state.comparative_metrics`
**Type** : Dictionnaire  
**Description** : Métriques comparatives (résultat de `generate_comparative_results()`)

**Contenu** : Voir section `generate_comparative_results()` pour la structure

---

#### `st.session_state.comparative_charts`
**Type** : Buffer d'image  
**Description** : Graphiques de performance comparatifs

**Génération** : Par `generate_comparative_charts()`

---

#### `st.session_state.comparative_trajectory`
**Type** : Buffer d'image  
**Description** : Visualisation comparative des trajectoires

**Génération** : Par `generate_trajectory_comparison()`

---

#### `st.session_state.comparative_report`
**Type** : String  
**Description** : Rapport textuel de performance

**Génération** : Par `generate_performance_report()`

---

#### `st.session_state.simple_comparative_config`
**Type** : Dictionnaire  
**Description** : Configuration utilisée pour la comparaison simplifiée

**Contenu** : Paramètres du formulaire de comparaison

---

## Modules highlight_plus

### Simulation

#### `highlight_plus/simulation/plume_model.py`

##### Classe `PlumeConfig`
**Description** : Configuration d'un panache de méthane

**Attributs** :
- `leak_x`, `leak_y` : Float - Position de la source (m)
- `leak_intensity` : Float - Débit de la fuite (kg/s)
- `wind_speed` : Float - Vitesse du vent (m/s)
- `wind_direction` : Float - Direction du vent (degrés)
- `sigma_x`, `sigma_y` : Float - Coefficients de dispersion (m)
- `decay_rate` : Float - Taux de décroissance temporelle (s⁻¹)

##### Classe `MethanePlume`
**Description** : Modèle de panache de méthane basé sur l'équation d'advection-diffusion gaussienne

**Méthodes principales** :
- `concentration(x, y, time)` : Calcule la concentration en un point
- `gradient(x, y, time)` : Calcule le gradient de concentration
- `create_concentration_map()` : Crée une carte de concentration

**Formule** :
```
C(x,y,t) = (Q / (2π σ_x σ_y u)) × exp(-((x-x₀')²/(2σ_x²) + (y-y₀')²/(2σ_y²))) × exp(-λt)
```

---

#### `highlight_plus/simulation/environment.py`

##### Classe `EnvironmentConfig`
**Description** : Configuration de l'environnement de simulation

**Attributs** :
- `world_size` : Tuple - Taille du monde (width, height) en mètres
- `time_step` : Float - Pas de temps (s)
- `max_steps` : Int - Nombre maximum d'étapes
- `base_power` : Float - Puissance de base (W)
- `speed_coefficient` : Float - Coefficient de vitesse (W/(m/s))
- `altitude_coefficient` : Float - Coefficient d'altitude (W/m)

##### Classe `MethaneDetectionEnv`
**Description** : Environnement Gymnasium pour la simulation

**Méthodes principales** :
- `reset()` : Réinitialise l'environnement
- `step(action, teacher)` : Exécute une action
- `_get_observation(teacher)` : Construit l'observation (16 dimensions)
- `_calculate_energy_cost(action)` : Calcule le coût énergétique

**Espace d'Observation** : 16 dimensions
- Position (x, y, z)
- Vitesse (vx, vy, vz)
- Concentration, détection
- Gradient (grad_x, grad_y)
- Vent (wind_x, wind_y)
- SNR
- Prédiction GP (μ, σ)
- Temps normalisé

**Espace d'Action** : 3 dimensions normalisées [-1, 1]
- Déplacement (dx, dy, dz)

---

### Capteurs

#### `highlight_plus/sensors/tdlas_sensor.py`

##### Classe `TDLASConfig`
**Description** : Configuration du capteur TDLAS

**Attributs** :
- `detection_threshold` : Float - Seuil de détection (kg/m³)
- `noise_level` : Float - Niveau de bruit (σ)
- `range_max`, `range_min` : Float - Portées (m)
- `laser_wavelength` : Float - Longueur d'onde (nm)
- `power` : Float - Puissance laser (mW)

##### Classe `TDLASSensor`
**Description** : Capteur TDLAS simulé

**Méthodes principales** :
- `measure_concentration(true_concentration, distance, ...)` : Mesure la concentration avec bruit
- `measure_at_position(x, y, z, plume_concentration, ...)` : Mesure à une position donnée

**Loi de Beer-Lambert** :
```
I_détectée = I₀ × exp(-α × C × L) × (ρ / h²)
```

**Modèle de Bruit** :
```
Bruit_total = Bruit_électronique + Bruit_atmosphérique + Bruit_interférence
```

---

### Modèles IA

#### `highlight_plus/models/teacher_gp.py`

##### Classe `TeacherConfig`
**Description** : Configuration du Teacher (GP)

**Attributs** :
- `kernel_length_scale` : Float - Échelle de longueur du noyau GP (m)
- `kernel_variance` : Float - Variance du noyau
- `exploration_parameter` : Float - Paramètre d'exploration (β pour UCB)
- `max_step_size`, `min_step_size` : Float - Tailles de pas (m)

##### Classe `GaussianProcessTeacher`
**Description** : Expert utilisant les Processus Gaussiens pour l'apprentissage actif

**Méthodes principales** :
- `add_observation(x, y, concentration)` : Ajoute une observation au GP
- `select_next_point(current_pos, estimated_source)` : Sélectionne le prochain point optimal
- `get_leak_position()` : Estime la position de la fuite
- `get_all_leak_positions()` : Retourne toutes les positions détectées
- `get_confidence_map()` : Génère une carte de confiance GP

**Noyau RBF** :
```
k(x, x') = variance × exp(-||x - x'||² / (2 × length_scale²))
```

**Fonction d'Acquisition UCB** :
```
UCB(x) = μ(x) + β × σ(x)
```

---

#### `highlight_plus/models/student_rl.py`

##### Classe `StudentConfig`
**Description** : Configuration du Student (RL)

**Attributs** :
- `learning_rate` : Float - Taux d'apprentissage
- `gamma` : Float - Facteur de discount (0.99)
- `epsilon_start`, `epsilon_end` : Float - Exploration (début/fin)
- `buffer_size` : Int - Taille du buffer d'expérience
- `batch_size` : Int - Taille du batch
- `hidden_layers` : List - Architecture du réseau ([256, 256, 128])
- `lambda_kl` : Float - Poids de distillation (0.1)

##### Classe `StudentRL`
**Description** : Apprenti utilisant l'apprentissage par renforcement

**Méthodes principales** :
- `select_action(state, teacher_guidance)` : Sélectionne une action
- `store_experience(state, action, reward, next_state, done)` : Stocke une expérience
- `learn()` : Effectue un pas d'apprentissage
- `update_target_network()` : Met à jour le réseau cible

**Architecture du Réseau** :
```
Entrée (16) → Couche 1 (256) → Couche 2 (256) → Couche 3 (128) → Sortie (3)
```

**Perte Totale** :
```
L_total = L_RL + λ × L_KL(π_teacher || π_student)
```

---

### Analyse

#### `highlight_plus/analysis/enhanced_detector.py`

##### Classe `DetectionEvent`
**Description** : Événement de détection

**Attributs** :
- `position` : Array - Position de détection
- `concentration` : Float - Concentration mesurée
- `confidence` : Float - Confiance de détection
- `step` : Int - Étape de détection
- `timestamp` : Float - Timestamp
- `is_valid` : Bool - Si la détection est valide

##### Classe `EnhancedDetector`
**Description** : Détecteur amélioré avec validation multi-critères

**Méthodes principales** :
- `validate_detection(position, concentration, gradient)` : Valide une détection
- `estimate_leak_position()` : Estime la position de la fuite
- `estimate_all_leak_positions(min_probability=0.75, min_distance=10.0)` : Retourne toutes les positions détectées
- `get_statistics()` : Retourne les statistiques

**Critères de Validation** :
1. Seuil de concentration adaptatif
2. Gradient cohérent
3. Stabilité temporelle
4. Clustering spatial
5. Validation GP probabiliste

---

#### `highlight_plus/analysis/methane_leak_validator.py`

##### Classe `MethaneLeakValidator`
**Description** : Validateur GP pour l'estimation probabiliste de la position de fuite

**Méthodes principales** :
- `add_measurement(x, y, concentration)` : Ajoute une mesure
- `get_leak_position()` : Estime la meilleure position de fuite
- `get_all_leak_positions(min_probability=0.75, min_distance=10.0)` : Retourne toutes les positions avec probabilité élevée
- `get_confidence_map()` : Génère une carte de confiance GP

**Méthode d'Estimation** :
1. Prédiction GP sur grille fine
2. Calcul de score combiné (70% concentration + 30% confiance)
3. Identification des candidats
4. Clustering DBSCAN
5. Filtrage strict (probabilité ≥ 75%)

---

#### `highlight_plus/analysis/performance_validator.py`

##### Classe `PerformanceMetrics`
**Description** : Métriques de performance

**Attributs** :
- `n_detections` : Int - Nombre de détections
- `first_detection_time` : Float - Temps de première détection (s)
- `first_detection_step` : Int - Étape de première détection
- `convergence_time` : Float - Temps de convergence (s)
- `mission_success` : Bool - Si la mission a réussi
- `overall_score` : Float - Score global (0-100)
- `localization_accuracy` : Object - Objet `LocalizationAccuracy`

##### Classe `LocalizationAccuracy`
**Description** : Précision de localisation

**Attributs** :
- `true_position` : Array - Position réelle
- `detected_position` : Array - Position détectée
- `error_distance` : Float - Erreur de distance (m)
- `error_angle` : Float - Erreur d'angle (degrés)
- `tolerance_radius` : Float - Rayon de tolérance (m)
- `is_within_tolerance` : Bool - Si dans la tolérance

##### Classe `PerformanceValidator`
**Description** : Validateur de performance

**Méthodes principales** :
- `add_detection(position, time, step)` : Ajoute une détection
- `compute_metrics(true_position, detected_position)` : Calcule les métriques
- `generate_report()` : Génère un rapport

---

## Flux de Données et Exécution

### Ordre d'Exécution Typique

1. **Configuration** :
   - L'utilisateur configure les paramètres via `show_configuration_tab()`
   - Les configurations sont stockées dans `st.session_state`

2. **Simulation** :
   - L'utilisateur lance une simulation via `run_simulation()`
   - La simulation utilise les configurations stockées
   - Les résultats sont stockés dans `st.session_state.simulation_results`

3. **Affichage des Résultats** :
   - Les résultats sont affichés via `display_performance_metrics()`
   - La trajectoire est visualisée via `display_trajectory_visualization()`

4. **Comparaison** :
   - L'utilisateur peut comparer avec la stratégie naïve via `show_comparative_simple_tab()`
   - Les résultats comparatifs sont générés et affichés

### Flux de Données

```
Configuration → Session State → Simulation → Results → Display
     ↓              ↓              ↓            ↓         ↓
  show_*()    st.session_    run_sim()    results   display_*()
              state.*
```

### Cycle de Simulation

```
1. OBSERVATION
   ├── Position du drone
   ├── Vitesse du drone
   ├── Concentration mesurée
   ├── Gradient de concentration
   └── Prédiction GP

2. PLANIFICATION (Teacher)
   ├── Mise à jour GP
   ├── Calcul UCB
   └── Sélection point optimal

3. NAVIGATION (Student)
   ├── Évaluation état
   ├── Action réseau
   └── Combinaison Teacher

4. EXÉCUTION
   ├── Déplacement
   ├── Consommation énergie
   └── Mesure concentration

5. APPRENTISSAGE
   ├── Mise à jour GP
   ├── Stockage expérience
   └── Mise à jour réseau

6. VALIDATION
   ├── Critères détection
   ├── Estimation position
   └── Clustering
```

---

## Notes Importantes

### Points d'Attention

1. **Multi-fuites** :
   - Le système supporte plusieurs positions de fuite via `MultiSourcePlume` et `st.session_state.leak_positions`
   - Le drone navigue vers la fuite la plus proche non détectée
   - La simulation continue après chaque détection
   - Toutes les positions détectées sont stockées dans `st.session_state.detected_leaks`

2. **Probabilité GP** :
   - Toutes les positions détectées sont filtrées avec un seuil minimum de 75%
   - Les positions sont triées par probabilité GP décroissante
   - La meilleure position (probabilité la plus élevée) est utilisée pour les statistiques

3. **Meilleure Position** :
   - La position avec la probabilité GP la plus élevée est utilisée pour :
     - Calcul des statistiques de performance
     - Affichage dans "Détails de Localisation"
     - Validation de la mission

4. **Toutes les Positions** :
   - Toutes les positions détectées (probabilité > 75%) sont stockées dans `st.session_state.detected_leaks`
   - Elles sont affichées dans "Position Estimée (GP Validator)"
   - Elles sont utilisées pour la détection multi-fuites

5. **Stratégie Adaptative** :
   - En mode Teacher-Student, les poids Teacher/Student sont ajustés dynamiquement
   - Formule : `Poids_Teacher = 0.8 - (0.5 × Confiance_Student)`
   - Le Student augmente son influence avec sa confiance

6. **Extraction GP** :
   - Les positions sont extraites de la carte GP toutes les 5 étapes
   - Clustering DBSCAN pour regrouper les candidats proches
   - Filtrage strict avec probabilité ≥ 75%

### Bonnes Pratiques

1. **Initialisation** :
   - Toujours initialiser les variables de session_state avant utilisation
   - Vérifier l'existence avec `if 'key' not in st.session_state`

2. **Performance** :
   - Les visualisations sont mises à jour toutes les 5 étapes pour éviter la surcharge
   - Les métriques sont calculées toutes les 10 étapes

3. **Erreurs** :
   - Gérer les erreurs avec `try-except` dans les fonctions critiques
   - Logger les erreurs pour le débogage

4. **Multi-fuites** :
   - Vérifier `use_multi_source` avant d'utiliser `MultiSourcePlume`
   - Gérer les cas où `leak_positions` est vide

---

## Version

**Document créé en** : Décembre 2025  
**Version du système** : 1.0.0  

---

## Contact

Pour toute question ou clarification sur cette documentation, veuillez consulter le code source ou contacter l'équipe de développement.

**Équipe HIGHLIGHT+** :
- Housséni YABRE - ETUDIANT en Informatique et Systèmes d'Information à l'UTT
- Kabinet SYLLA - ETUDIANT en Informatique et Systèmes d'Information à l'UTT
- Nobert Bassooma DIDANERA - Etudiant en fin de Master IA et Big Data, En mobilité à l'UTT

**Email** : housseni.yabre@utt.fr , kabinet.sylla@utt.fr , bassooma_norbert.didanera@utt.fr

---

*Documentation complète et structurée du système HIGHLIGHT+ - Version 1.0.0*
