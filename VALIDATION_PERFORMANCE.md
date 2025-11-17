# Système de Validation de Performance

## Vue d'ensemble

Un système complet de validation de performance a été ajouté à HIGHLIGHT+ pour mesurer précisément l'efficacité des méthodes de détection. Ce système compare systématiquement les positions réelles des fuites avec les détections effectuées et calcule des métriques détaillées.

## Fonctionnalités Principales

### 1. **Comparaison Position Réelle vs Position Détectée**

Le système connaît la position réelle de la fuite (définie dans la configuration du panache) et compare toutes les détections avec cette position de référence.

**Métriques calculées :**
- **Distance d'erreur** : Distance euclidienne entre la position détectée et la position réelle
- **Angle d'erreur** : Direction de l'erreur de localisation (en degrés)
- **Détection dans la tolérance** : Vérifie si la détection est dans le rayon de tolérance (10 m par défaut)

### 2. **Temps de Détection**

Le système mesure le temps nécessaire pour détecter la fuite :
- **Première détection** : Étape et temps de la première détection
- **Temps de convergence** : Temps nécessaire pour une détection valide (dans la tolérance)
- **Performance temporelle** : Évaluation de la rapidité de détection

### 3. **Précision de Localisation**

Calcul automatique de la précision :
- **Meilleure détection** : Détection la plus proche de la source réelle
- **Distance moyenne** : Distance moyenne de toutes les détections à la source
- **Score de localisation** : Score sur 100 basé sur la précision

### 4. **Métriques de Performance Globales**

Le système calcule des scores globaux pour évaluer la performance :

#### Score de Détection (0-100)
- Basé sur la rapidité de détection
- Basé sur le taux de détection
- Plus rapide = score plus élevé

#### Score de Localisation (0-100)
- Basé sur la distance d'erreur
- Score maximal si erreur < tolérance
- Décroît avec l'augmentation de l'erreur

#### Score d'Efficacité (0-100)
- Basé sur l'énergie consommée par détection
- Idéal : < 100 J/détection
- Évalue l'efficacité énergétique

#### Score Global (0-100)
- **Moyenne pondérée** :
  - 40% Score de Détection
  - 40% Score de Localisation
  - 20% Score d'Efficacité

## Utilisation dans l'Interface Streamlit

### Affichage des Métriques

Dans l'onglet **"Résultats & Métriques"**, une nouvelle section **"Validation de Performance"** affiche :

1. **Métriques Principales** :
   - Score Global
   - Temps de Détection (avec badge de statut)
   - Précision de Localisation
   - Statut de Mission (Reussie ou Partielle)

2. **Détails de Localisation** :
   - Position Réelle (x, y)
   - Position Détectée (x, y)
   - Erreur (distance et angle)

3. **Scores Détaillés** :
   - Score Détection avec barre de progression
   - Score Localisation avec barre de progression
   - Score Efficacité avec barre de progression

### Visualisation Interactive

La visualisation de trajectoire a été améliorée pour afficher :

1. **Position Réelle de la Fuite** : Marqueur rouge (X) indiquant la position exacte de la fuite

2. **Zone de Tolérance** : Cercle en pointillés montrant la zone acceptable autour de la fuite (rayon de 10 m par défaut)

3. **Meilleure Détection** : Marqueur vert (diamant) indiquant la détection la plus proche de la source

4. **Ligne d'Erreur** : Ligne orange pointillée reliant la position réelle à la meilleure détection, avec la distance affichée

### Logs Détaillés

Pendant la simulation, les logs affichent maintenant :
- `MISSION REUSSIE - Score global: XX.X/100` (si mission réussie)
- `MISSION PARTIELLE - Score: XX.X/100` (si mission partielle)
- Détails de première détection
- Erreur de localisation

## Architecture Technique

### Module `performance_validator.py`

#### Classes Principales

1. **`PerformanceValidator`** :
   - Gère la validation de performance
   - Calcule toutes les métriques
   - Génère les rapports

2. **`PerformanceMetrics`** :
   - Structure de données pour toutes les métriques
   - Contient : détection, localisation, temps, énergie, scores

3. **`LocalizationAccuracy`** :
   - Détails de précision de localisation
   - Position réelle, position détectée, erreur

4. **`DetectionResult`** :
   - Résultat d'une détection individuelle
   - Position, concentration, étape, temps

### Intégration avec la Simulation

Le validateur est initialisé au début de chaque simulation avec :
- Position réelle de la fuite (depuis `plume_config`)
- Rayon de tolérance (10 m par défaut)
- Pas de temps de simulation

À chaque détection, le système :
1. Enregistre la détection dans le validateur
2. Met à jour l'énergie consommée
3. Calcule les métriques en temps réel

À la fin de la simulation :
1. Calcule toutes les métriques finales
2. Génère un rapport détaillé
3. Affiche les résultats dans l'interface

## Export des Résultats

L'export JSON inclut maintenant :
- Toutes les métriques de performance
- Rapport détaillé de validation
- Positions réelles et détectées
- Scores de performance
- Statut de mission

Format du fichier exporté :
```json
{
  "performance_metrics": {
    "n_detections": 5,
    "first_detection_time": 12.5,
    "localization_accuracy": {
      "error_distance": 3.2,
      "is_within_tolerance": true,
      ...
    },
    "overall_score": 85.3,
    "mission_success": true,
    ...
  },
  "performance_report": {
    "summary": {...},
    "detection": {...},
    "localization": {...},
    ...
  }
}
```

## Configuration

### Rayon de Tolérance

Par défaut : **10 mètres**

Peut être modifié dans `streamlit_app.py` :
```python
validator = PerformanceValidator(
    true_leak_position=true_leak_pos,
    tolerance_radius=10.0,  # Modifier ici
    time_step=env_config.time_step
)
```

## Interprétation des Scores

### Score Global

- **80-100** : Excellent - Mission très réussie
- **60-79** : Bon - Mission réussie avec améliorations possibles
- **40-59** : Acceptable - Mission partielle
- **0-39** : Insuffisant - Mission échouée

### Temps de Détection

- **< 10 s** : Rapide
- **10-30 s** : Acceptable
- **> 30 s** : Lent

### Précision de Localisation

- **Erreur ≤ Tolérance** : Précis
- **Erreur > Tolérance** : Imprécis

## Exemples d'Utilisation

### Vérifier si une détection est correcte

```python
metrics = validator.compute_metrics()
if metrics.mission_success:
    print("Mission reussie !")
    print(f"Erreur de localisation : {metrics.localization_accuracy.error_distance:.2f}m")
```

### Exporter les résultats

```python
report = validator.generate_report(metrics)
validator.export_results(metrics, "results.json")
```

## Avantages du Système

1. **Validation Automatique** : Compare automatiquement les détections avec la réalité
2. **Métriques Complètes** : Mesure tous les aspects de la performance
3. **Visualisation Claire** : Interface graphique montrant position réelle vs détectée
4. **Scores Standardisés** : Scores sur 100 pour faciliter la comparaison
5. **Export Détaillé** : Rapports JSON complets pour analyse approfondie

## Améliorations Futures Possibles

- [ ] Support de multiples fuites simultanées
- [ ] Tests statistiques de significativité
- [ ] Comparaison entre différentes méthodes (Teacher vs Student vs Baseline)
- [ ] Analyse de robustesse sur plusieurs runs
- [ ] Métriques de confiance pour les détections
