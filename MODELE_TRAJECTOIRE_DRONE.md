# Modélisation de la Trajectoire du Drone dans HIGHLIGHT+

## Vue d'ensemble

La trajectoire du drone est modélisée de manière **discrète** avec un modèle de mouvement basé sur des **actions normalisées** converties en déplacements réels. Le système utilise une approche **temps réel** avec des pas de temps fixes.

---

## 1. Espace d'Action

### Définition
L'espace d'action est un **espace continu** normalisé :

```python
Action Space: Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
```

- **3 dimensions** : `[Δx, Δy, Δz]` (déplacement normalisé)
- **Valeurs** : Entre -1.0 et 1.0 (normalisées)
- **Signification** :
  - `action[0]` : Déplacement horizontal X (normalisé)
  - `action[1]` : Déplacement horizontal Y (normalisé)
  - `action[2]` : Déplacement vertical Z/altitude (normalisé)

---

## 2. Conversion Action → Déplacement Réel

### Méthode : `_action_to_displacement()`

```python
def _action_to_displacement(self, action: np.ndarray) -> np.ndarray:
    """Convertit l'action normalisée en déplacement réel"""
    max_displacement = self.config.max_speed * self.config.time_step
    displacement = action * max_displacement
    return displacement
```

### Formule Mathématique

**Déplacement réel** = **Action normalisée** × **Déplacement maximal**

Où :
- **Déplacement maximal** = `max_speed` × `time_step`
- **Exemple** : Si `max_speed = 5.0 m/s` et `time_step = 0.1 s`
  - `max_displacement = 5.0 × 0.1 = 0.5 m`
  - Une action `[1.0, 0.0, 0.0]` → déplacement de `[0.5 m, 0.0 m, 0.0 m]`

---

## 3. Mise à Jour de la Position

### Processus dans `step()`

```python
# 1. Conversion action → déplacement
displacement = self._action_to_displacement(action)

# 2. Sauvegarde de l'ancienne position
old_position = self.drone_position.copy()

# 3. Mise à jour de la position
self.drone_position += displacement

# 4. Application des contraintes
self._apply_constraints()

# 5. Calcul de la vitesse
self.drone_velocity = (self.drone_position - old_position) / self.config.time_step

# 6. Enregistrement dans la trajectoire
self.trajectory.append(self.drone_position.copy())
```

### Formule de Mise à Jour

**Position(t+1)** = **Position(t)** + **Déplacement**

**Vitesse** = **ΔPosition** / **time_step**

---

## 4. Contraintes Physiques

### Méthode : `_apply_constraints()`

```python
def _apply_constraints(self):
    """Applique les contraintes physiques du drone"""
    # Contraintes spatiales (limites du monde)
    self.drone_position[0] = np.clip(self.drone_position[0], 0, self.world_width)
    self.drone_position[1] = np.clip(self.drone_position[1], 0, self.world_height)
    
    # Contrainte d'altitude
    self.drone_position[2] = np.clip(
        self.drone_position[2], 
        self.config.min_altitude, 
        self.config.max_altitude
    )
```

### Contraintes Appliquées

1. **Limites spatiales horizontales** :
   - X : `[0, world_width]` (par défaut : `[0, 100]` m)
   - Y : `[0, world_height]` (par défaut : `[0, 100]` m)

2. **Limites d'altitude** :
   - Z : `[min_altitude, max_altitude]` (par défaut : `[2.0, 20.0]` m)

3. **Méthode** : Utilisation de `np.clip()` pour forcer les valeurs dans les limites

---

## 5. Stockage de la Trajectoire

### Structure de Données

```python
self.trajectory = [self.drone_position.copy()]  # Initialisation
# ...
self.trajectory.append(self.drone_position.copy())  # À chaque step
```

### Format

- **Type** : Liste de tableaux numpy `(3,)`
- **Chaque élément** : `[x, y, z]` en mètres
- **Exemple** :
  ```python
  trajectory = [
      [10.0, 10.0, 5.0],   # Position initiale
      [10.5, 10.0, 5.0],   # Après step 1
      [11.0, 10.2, 5.0],   # Après step 2
      ...
  ]
  ```

---

## 6. Paramètres de Configuration

### Paramètres Clés (EnvironmentConfig)

```python
@dataclass
class EnvironmentConfig:
    # Dimensions du monde
    world_size: Tuple[float, float] = (100.0, 100.0)  # m
    
    # Paramètres temporels
    time_step: float = 0.1  # s (pas de temps)
    max_steps: int = 1000   # Nombre maximum d'étapes
    
    # Conditions initiales
    initial_position: Tuple[float, float] = (10.0, 10.0)  # m
    initial_altitude: float = 5.0  # m
    
    # Contraintes du drone
    max_speed: float = 5.0      # m/s
    max_altitude: float = 20.0  # m
    min_altitude: float = 2.0   # m
```

---

## 7. Calcul de la Vitesse

### Formule

```python
self.drone_velocity = (self.drone_position - old_position) / self.config.time_step
```

### Signification

- **Vitesse** = **Déplacement** / **Durée du pas de temps**
- **Unité** : m/s
- **Exemple** : Si déplacement de `[0.5, 0.2, 0.0]` m en `0.1` s
  - Vitesse = `[5.0, 2.0, 0.0]` m/s

---

## 8. Exemple de Trajectoire Complète

### Simulation sur 3 Steps

**Configuration** :
- `max_speed = 5.0 m/s`
- `time_step = 0.1 s`
- `max_displacement = 0.5 m`

**Actions** :
- Step 1 : `action = [1.0, 0.0, 0.0]` → déplacement `[0.5, 0.0, 0.0]` m
- Step 2 : `action = [0.0, 1.0, 0.0]` → déplacement `[0.0, 0.5, 0.0]` m
- Step 3 : `action = [-0.5, 0.5, 0.0]` → déplacement `[-0.25, 0.25, 0.0]` m

**Trajectoire Résultante** :
```
Position initiale : [10.0, 10.0, 5.0]
Step 1           : [10.5, 10.0, 5.0]  (vitesse: [5.0, 0.0, 0.0] m/s)
Step 2           : [10.5, 10.5, 5.0]  (vitesse: [0.0, 5.0, 0.0] m/s)
Step 3           : [10.25, 10.75, 5.0] (vitesse: [-2.5, 2.5, 0.0] m/s)
```

---

## 9. Intégration avec les Agents IA

### Mode Simple
- Actions calculées directement à partir de la direction vers la cible et du gradient

### Mode Teacher (GP)
- Le Teacher suggère un point suivant via `select_next_point()`
- Conversion en action normalisée pour le déplacement

### Mode Teacher-Student (RL)
- Le Student (RL) génère des actions via `select_action()`
- Les actions peuvent être mélangées avec les suggestions du Teacher
- Apprentissage progressif de la politique de navigation

---

## 10. Caractéristiques du Modèle

### ✅ Avantages

1. **Simplicité** : Modèle discret facile à implémenter
2. **Contrôle précis** : Actions normalisées permettent un contrôle fin
3. **Contraintes intégrées** : Limites physiques appliquées automatiquement
4. **Temps réel** : Pas de temps fixe pour simulation en temps réel
5. **Traçabilité** : Trajectoire complète stockée pour analyse

### ⚠️ Limitations

1. **Modèle simplifié** : Pas de dynamique complexe (accélération, inertie)
2. **Pas de vent** : Le vent n'affecte pas directement le mouvement (seulement le panache)
3. **Mouvement instantané** : Pas de délai de réponse du drone

---

## 11. Utilisation pour la Visualisation

La trajectoire stockée dans `self.trajectory` est utilisée pour :

1. **Visualisation en temps réel** : Affichage de la trajectoire pendant la simulation
2. **Analyse comparative** : Comparaison entre différentes stratégies (Naïve vs HIGHLIGHT+)
3. **Métriques de performance** : Calcul de la distance parcourue, efficacité énergétique
4. **Cartes de confiance GP** : Superposition de la trajectoire sur les cartes de concentration

---

## Conclusion

Le modèle de trajectoire du drone dans HIGHLIGHT+ est un **modèle discret basé sur des actions normalisées**, avec conversion en déplacements réels et application de contraintes physiques. Il permet une simulation efficace et contrôlable, adaptée à l'apprentissage par renforcement et à l'optimisation de trajectoires pour la détection de fuites de méthane.

