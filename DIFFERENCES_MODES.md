# Différences entre les Modes de Simulation

## Vue d'Ensemble

HIGHLIGHT+ propose trois modes de simulation pour la détection de fuites de méthane (visibles dans l'interface utilisateur) :

1. **Mode Simple** (`simple`) : Mode basique avec stratégie multi-phase
2. **Mode Teacher** (`teacher_student`) : Mode utilisant uniquement l'Expert (Teacher) avec stratégie adaptative
3. **Mode Teacher-Student** (`full_learning`) : Mode utilisant l'Expert (Teacher) + l'Apprenti (Student) avec stratégie adaptative

**Note** : Les noms entre parenthèses sont les valeurs internes utilisées dans le code. Les noms affichés dans l'interface utilisateur sont "Mode Simple", "Mode Teacher" et "Mode Teacher-Student".

---

## Mode Simple (`simple`) - Stratégie Multi-Phase Basique

**Architecture :**
- Utilise une navigation directe basée sur la direction vers la cible
- Stratégie multi-phase simple sans IA
- **N'utilise PAS** le Teacher (GP) ni le Student (RL)
- **N'utilise PAS** le Validateur GP

**Fonctionnement :**
1. Navigation directe vers la position réelle de la fuite (connue dans la simulation)
2. Stratégie multi-phase optimisée :

   **Phase 1 (>25m) : Navigation rapide**
   - Direction directe vers la cible : `target_dir = vec_to_target / distance_to_target`
   - Action : `[target_dir[0] × 1.0, target_dir[1] × 1.0, 0.0]`
   - Vitesse maximale vers la cible
   
   **Phase 2 (10-25m) : Approche guidée**
   - Combinaison de direction cible (60%) + gradient (40%)
   - Si gradient disponible : `combined = 0.6 × target_dir + 0.4 × grad_dir`
   - Sinon : `combined = 0.8 × target_dir`
   - Action : `[combined[0] × 0.9, combined[1] × 0.9, 0.0]`
   
   **Phase 3 (<10m) : Recherche locale**
   - Si gradient disponible :
     - Combinaison : Gradient (50%) + Spirale (30%) + Direction cible (20%)
     - Mouvement en spirale autour de la source avec angle : `angle_to_source + (step × 0.3) % (2π)`
     - Action : `[combined[0] × 0.6, combined[1] × 0.6, 0.0]`
   - Sinon :
     - Mouvement circulaire tangentiel
     - Action : `[tangent_dir[0] × 0.5, tangent_dir[1] × 0.5, 0.0]`

3. Utilise le gradient de concentration pour guider la navigation dans les phases 2 et 3

**Avantages :**
- **Simplicité** : Implémentation directe, facile à comprendre
- **Rapidité** : Pas de calculs complexes d'IA
- **Stabilité** : Comportement déterministe

**Inconvénients :**
- **Performance limitée** : Pas d'optimisation intelligente
- **Pas d'apprentissage** : Ne s'améliore pas avec l'expérience
- **Sensibilité au bruit** : Peut être affecté par les conditions environnementales

---

## Mode Teacher (`teacher_student`) - Expert Seul

**Architecture :**
- Utilise uniquement le **Teacher (Expert GP)**
- N'utilise PAS le Student (Apprenti RL)
- Stratégie adaptative basée sur la confiance GP du Validateur
- **Validateur GP actif** : Estimation probabiliste de la position de fuite

**Fonctionnement :**
1. Le Teacher (Processus Gaussiens) guide directement la navigation
2. Le Validateur GP fournit une estimation probabiliste de la position de fuite
3. Stratégie multi-phase optimisée avec utilisation intensive du GP Validator :

   **Phase 1 (>25m) : Navigation rapide**
   - Si estimation GP avec confiance > 60% : Utilise GP (70%) + direction réelle (30%)
   - Sinon : Navigation directe vers la cible réelle
   - Direction calculée via `teacher.select_next_point()` avec UCB
   
   **Phase 2 (10-25m) : Approche guidée**
   - Récupération de l'estimation GP (seuil réduit à 30% pour utilisation précoce)
   - Poids adaptatifs selon la confiance GP :
     - Confiance > 70% : GP (50%) + Teacher (30%) + Gradient (20%)
     - Confiance 50-70% : GP (35%) + Direct (30%) + Teacher (25%) + Gradient (10%)
     - Confiance < 50% : Direct (45%) + Gradient (35%) + Teacher (20%)
   - Utilisation de `teacher.select_next_point()` avec estimation GP comme cible
   
   **Phase 3 (<10m) : Recherche locale**
   - Récupération de l'estimation GP (seuil très bas à 25% pour recherche locale)
   - Si pas d'estimation GP : Utilise position réelle comme fallback
   - Poids adaptatifs selon la confiance GP :
     - Confiance > 70% : GP (60%) + Gradient (25%) + Teacher (15%)
     - Confiance 50-70% : GP (45%) + Gradient (30%) + Teacher (15%) + Spirale (10%)
     - Confiance < 50% : Gradient (40%) + Spirale (30%) + Teacher (20%) + GP (10%)
   - Recherche en spirale autour de l'estimation GP ou position réelle

**Avantages :**
- **Performance immédiate** : Le Teacher est déjà expert, pas besoin d'apprentissage
- **Stable et prévisible** : Comportement cohérent à chaque exécution
- **Rapide** : Pas de temps d'apprentissage, détection directe
- **Optimal pour détection unique** : Parfait pour une seule fuite
- **Validateur GP actif** : Estimation probabiliste de la position

**Inconvénients :**
- **Pas d'amélioration** : Le système ne s'améliore pas avec l'expérience
- **Pas d'adaptation fine** : Ne s'adapte pas aux patterns spécifiques de l'environnement
- **Consommation GP** : Calculs GP peuvent être coûteux

---

## Mode Teacher-Student (`full_learning`) - Expert + Apprenti

**Architecture :**
- Utilise le **Teacher (Expert GP)** pour guidance stratégique
- Utilise le **Student (Apprenti RL)** pour navigation tactique
- **Stratégie adaptative** : Mélange dynamique Teacher/Student selon la confiance

**Fonctionnement :**
1. **Au début** (Student non entraîné) :
   - Teacher : **80%** d'influence
   - Student : **20%** d'influence
   - Le Student apprend en observant le Teacher via distillation de connaissance

2. **Progressivement** (Student apprend) :
   - La confiance du Student augmente avec la qualité de son apprentissage
   - Confiance calculée : `Confiance_Student = max(0, min(1, 1 - avg_loss/0.5))`
   - Les poids s'ajustent dynamiquement :
     - Teacher : **80% → 30%** (diminue)
     - Student : **20% → 70%** (augmente)

3. **À la fin** (Student bien entraîné) :
   - Teacher : **30%** d'influence (guidance stratégique)
   - Student : **70%** d'influence (navigation optimisée)
   - Le Student devient autonome et rapide

**Stratégie Multi-Phase Adaptative :**

**Phase 1 (>25m) : Navigation rapide**
- Utilise l'estimation GP si disponible (seuil 30%), sinon position réelle
- Mélange adaptatif : `teacher_weight × teacher_dir + student_weight × student_action + 0.2 × nav_dir`
- Direction Teacher calculée via `teacher.select_next_point()` avec estimation GP
- Action Student générée via `student.select_action()` avec guidance Teacher

**Phase 2 (10-25m) : Approche guidée**
- Utilise Teacher avec estimation GP pour convergence guidée
- Si gradient disponible :
  - Mélange : `teacher_weight × teacher_dir + student_weight × student_action + 0.25 × grad_dir + 0.15 × center_dir`
- Sinon :
  - Mélange : `teacher_weight × teacher_dir + student_weight × student_action + 0.2 × center_dir`
- Centre de recherche : Estimation GP si disponible, sinon position réelle

**Phase 3 (<10m) : Recherche locale**
- Récupération de l'estimation GP (seuil 40% pour recherche locale)
- Si gradient disponible :
  - Mouvement en spirale autour du centre estimé (GP ou réel)
  - Mélange : `teacher_weight × teacher_dir + student_weight × student_action + 0.25 × grad_dir + 0.15 × circular_dir + 0.1 × center_dir`
- Sinon :
  - Mouvement circulaire tangentiel
  - Mélange : `teacher_weight × teacher_dir + student_weight × student_action + 0.2 × tangent_dir + 0.15 × center_dir`

**Avantages :**
- **Apprentissage progressif** : Le Student s'améliore avec l'expérience
- **Performance optimale** : Combine expertise Teacher + efficacité Student
- **Adaptation** : S'adapte aux patterns spécifiques de l'environnement
- **Efficacité énergétique** : Le Student apprend à optimiser les trajectoires
- **Optimal pour détections multiples** : Peut apprendre à détecter plusieurs fuites
- **Validateur GP actif** : Estimation probabiliste avec détection multi-fuites

**Inconvénients :**
- **Démarrage plus lent** : Nécessite quelques étapes pour que le Student apprenne
- **Variabilité initiale** : Performance peut varier au début
- **Complexité** : Plus de composants à gérer

---

## Comparaison des Performances

| Critère | Mode Simple | Mode Teacher | Mode Teacher-Student |
|---------|-------------|--------------|---------------------|
| **Performance initiale** | Bonne | Excellente | Bonne (améliore avec le temps) |
| **Performance finale** | Bonne | Excellente | Optimale (après apprentissage) |
| **Vitesse de détection** | Rapide | Très rapide | Rapide (après quelques étapes) |
| **Efficacité énergétique** | Moyenne | Bonne | Excellente (après apprentissage) |
| **Adaptation** | Limitée | Limitée | Excellente |
| **Stabilité** | Très stable | Très stable | Stable (après apprentissage) |
| **Complexité** | Faible | Moyenne | Élevée |

---

## Quand Utiliser Chaque Mode ?

### Utilisez **Mode Simple** si :
- Vous voulez une implémentation basique et rapide
- Vous testez le système pour la première fois
- Vous avez besoin d'un comportement déterministe simple
- Les performances de base suffisent

### Utilisez **Mode Teacher** si :
- Vous voulez une **détection immédiate et fiable**
- Vous avez besoin de **résultats cohérents** à chaque exécution
- Vous détectez **une seule fuite**
- Vous préférez la **stabilité** à l'adaptation
- Vous voulez les meilleures performances sans apprentissage

### Utilisez **Mode Teacher-Student** si :
- Vous voulez **maximiser les performances** à long terme
- Vous détectez **plusieurs fuites** (le Student apprend les patterns)
- Vous voulez **optimiser l'efficacité énergétique**
- Vous avez le temps pour quelques étapes d'apprentissage initial
- Vous voulez un système qui s'améliore avec l'expérience

---

## Détails Techniques

### Calcul de la Confiance du Student

La confiance du Student est calculée à partir de la perte d'apprentissage moyenne :

```
Confiance_Student = max(0.0, min(1.0, 1.0 - avg_loss / 0.5))
```

où :
- `avg_loss` est la **perte RL moyenne sur les 10 dernières itérations** (`student.loss_history[-10:]`)
- Une perte faible (< 0.1) indique une bonne performance et donc une confiance élevée
- Si moins de 10 itérations d'apprentissage : `Confiance_Student = 0.1` (favoriser Teacher au début)
- La confiance est recalculée à chaque étape d'apprentissage

### Poids Adaptatifs Teacher/Student

Les poids Teacher/Student sont ajustés dynamiquement à chaque étape :

```
teacher_weight = 0.8 - (0.5 × student_confidence)
student_weight = 0.2 + (0.5 × student_confidence)
```

Cette formulation garantit :
- **En début de mission** (Confiance_Student ≈ 0.1) : 
  - `teacher_weight ≈ 0.75`, `student_weight ≈ 0.25` (Teacher domine)
- **Progressivement** (Confiance_Student augmente) :
  - Les poids s'ajustent linéairement
- **En fin de mission** (Confiance_Student ≈ 1.0) : 
  - `teacher_weight = 0.3`, `student_weight = 0.7` (Student domine)
- Les poids sont recalculés à chaque étape d'apprentissage

### Action Finale

L'action finale combine les recommandations du Teacher et du Student, avec des composantes additionnelles selon la phase :

**Phase 1 (>25m) :**
```
a_final = w_Teacher × a_Teacher + w_Student × a_Student + 0.2 × nav_dir
```

**Phase 2 (10-25m) :**
```
a_final = w_Teacher × a_Teacher + w_Student × a_Student + 0.25 × grad_dir + 0.15 × center_dir
```
(si gradient disponible, sinon sans gradient)

**Phase 3 (<10m) :**
```
a_final = w_Teacher × a_Teacher + w_Student × a_Student + 0.25 × grad_dir + 0.15 × circular_dir + 0.1 × center_dir
```
(si gradient disponible, sinon avec mouvement tangentiel)

### Validateur GP et Détection Multi-Fuites

Les **Mode Teacher** et **Mode Teacher-Student** utilisent le **Validateur GP** (`MethaneLeakValidator`) pour :

1. **Estimation probabiliste de la position de fuite** :
   - Le Validateur GP utilise un **GP séparé** (différent du Teacher GP) pour modéliser la carte de concentration
   - Accumule toutes les mesures de concentration au fil du temps
   - Calcule un **score combiné** pour chaque position candidate :
     ```
     score = 0.7 × concentration_normalisée + 0.3 × confiance
     ```
     où `confiance = 1 - incertitude_normalisée`
   - Applique une pénalité si l'incertitude relative > 50% de la concentration
   - Retourne la position estimée avec sa probabilité (0.0 à 1.0)

2. **Seuils adaptatifs selon le nombre de mesures** :
   - **< 10 mesures** : Seuil réduit (threshold - 0.15, minimum 0.6) pour détection précoce
   - **≥ 10 mesures** : Seuil normal (par défaut 0.95)
   - **Phase 1** : Utilisé si probabilité > 60% (seuil élevé pour navigation rapide)
   - **Phase 2** : Utilisé si probabilité > 30% (seuil réduit pour approche guidée)
   - **Phase 3** : Utilisé si probabilité > 25% (seuil très bas pour recherche locale)

3. **Utilisation dans la navigation** :
   - Les poids dans le mélange de directions s'ajustent selon la probabilité GP
   - L'estimation GP est utilisée comme cible de navigation si disponible
   - Si pas d'estimation GP, fallback sur la position réelle

4. **Détection Multi-Fuites** :
   - Méthode `get_all_leak_positions(min_probability=0.75, min_distance=5.0)`
   - Détection de toutes les positions avec probabilité GP ≥ 75% (seuil strict)
   - Clustering manuel (pas DBSCAN) : regroupe les positions proches (< 5m)
   - Pour chaque groupe, garde la position avec la plus haute probabilité
   - Tri par probabilité décroissante
   - Retour de toutes les positions détectées (maximum 5 sources)
   - En mode multi-fuites, la simulation continue après chaque détection

---

## Améliorations Récentes

Pour optimiser les performances, nous avons implémenté :

1. **Stratégie Adaptative** :
   - Le Teacher domine au début (80%)
   - Le Student augmente progressivement (20% → 70%)
   - Confiance calculée à partir de la perte d'apprentissage

2. **Apprentissage Accéléré** :
   - `learning_starts` configurable (par défaut 1000)
   - Apprentissage à chaque étape après `learning_starts` expériences stockées
   - Mise à jour périodique du réseau cible (tous les 100 steps par défaut)

3. **Guidance Teacher** :
   - Le Student reçoit des suggestions du Teacher
   - Exploration guidée au lieu d'aléatoire
   - Distillation de connaissance active

4. **Mélange Adaptatif Multi-Phase** :
   - Toutes les phases utilisent le mélange adaptatif Teacher/Student
   - Performance au moins équivalente à **Mode Teacher** dès le début
   - Navigation guidée par estimation GP dans toutes les phases
   - Poids adaptatifs calculés à chaque étape selon la confiance du Student

5. **Validateur GP Intégré** :
   - Utilisation intensive du Validateur GP dans toutes les phases
   - Seuils adaptatifs selon la phase (60% Phase 1, 30% Phase 2, 25% Phase 3)
   - Poids dans le mélange de directions ajustés selon la confiance GP
   - Support complet pour la détection multi-fuites

---

## Détails Techniques du Teacher GP

### Fonction `select_next_point()`

Le Teacher GP utilise une stratégie multi-niveaux pour sélectionner le prochain point :

1. **Si `estimated_source` fourni (convergence fine)** :
   - **< 5m de la source estimée** : Recherche locale en spirale
     - 60% mouvement tangentiel + 40% mouvement radial
     - Pas adaptatif : 0.2m à 0.5m selon distance
   - **5-15m de la source estimée** : Convergence guidée
     - Pas adaptatif : 0.2m à 2.25m selon distance
     - Si gradient disponible : 70% direction source + 30% gradient
     - Sinon : Direction directe vers source estimée

2. **Si gradient significatif disponible** :
   - Direction suivant le gradient (vers la source)
   - Pas adaptatif selon la magnitude du gradient (max 2x si gradient fort)
   - Taille de pas : entre `min_step_size` (1.0m) et `max_step_size` (5.0m)

3. **Si peu d'observations (< 10) et `target_position` fourni** :
   - Navigation vers la cible avec exploration aléatoire (±0.3)
   - Pas adaptatif : min(max_step_size, distance × 0.5)

4. **Sinon (exploration active avec acquisition function)** :
   - Résolution de grille adaptative :
     - < 20 observations : 150×150 (très fine)
     - 20-50 observations : 120×120 (fine)
     - ≥ 50 observations : 100×100 (standard)
   - Fonction d'acquisition combinée :
     ```
     combined_acquisition = 0.6 × acquisition_values + 0.4 × uncertainty_norm
     ```
   - Contrainte de distance : entre `min_step_size` (1.0m) et `max_step_size` (5.0m)

### Fonction d'Acquisition UCB Améliorée

```
exploration_weight = max(0.3, 1.0 - n_obs / 50.0)  # De 1.0 à 0.3
exploitation_weight = 1.0 - exploration_weight
acquisition = exploitation_weight × mean_norm + exploration_weight × β × std_norm
```

où :
- `mean_norm` : Concentration normalisée (0-1)
- `std_norm` : Incertitude normalisée (0-1)
- `β` : Paramètre d'exploration (par défaut 2.0)
- Plus d'observations → plus d'exploitation, moins d'exploration

---

## Détails d'Implémentation

### Initialisation des Composants

**Mode Simple** (`simple`) :
- Aucun composant IA initialisé
- Utilise uniquement les calculs de gradient et de direction

**Mode Teacher** (`teacher_student`) :
- Teacher (GP) initialisé avec configuration par défaut
- Validateur GP activé (`use_gp_validator = True`)
- Student (RL) **NON initialisé**

**Mode Teacher-Student** (`full_learning`) :
- Teacher (GP) initialisé avec configuration par défaut
- Student (RL) initialisé avec distillation de connaissance depuis le Teacher
- Validateur GP activé (`use_gp_validator = True`)

### Mise à Jour des Composants

**Teacher (GP) :**
- Mise à jour à chaque étape avec nouvelle observation : `teacher.add_observation(x, y, concentration)`
- Le GP est **réentraîné à chaque nouvelle observation** (méthode `_update_gp()` appelée automatiquement)
- Minimum 2 observations requises pour entraîner le GP
- Kernel composite : `ConstantKernel × RBF + WhiteKernel`
- Fonction d'acquisition UCB avec poids adaptatifs :
  ```
  exploration_weight = max(0.3, 1.0 - n_obs / 50.0)  # De 1.0 à 0.3
  exploitation_weight = 1.0 - exploration_weight
  acquisition = exploitation_weight × mean_norm + exploration_weight × β × std_norm
  ```

**Student (RL) :**
- Stockage d'expérience : `student.store_experience(state, action, reward, next_state, done)`
- Apprentissage : `student.learn()` appelé **à chaque étape** après `learning_starts` expériences stockées (par défaut 1000)
- Mise à jour du réseau cible tous les `target_update_freq` steps (par défaut 100)
- Exploration guidée : Si `teacher_guidance` disponible et exploration (epsilon), utilise guidance Teacher avec bruit
- Si Student peu entraîné (< 50 itérations) et guidance disponible : `action = 0.7 × action + 0.3 × teacher_guidance`
- Décroissance adaptative de epsilon selon la perte récente

**Validateur GP :**
- Ajout de mesure à chaque étape : `gp_validator.add_measurement(position, concentration)`
- Le GP du Validateur est **réentraîné à chaque nouvelle mesure** (minimum 2 mesures)
- Estimation de position : `gp_validator.get_leak_position()` appelée selon les besoins de la phase
- Détection multi-fuites : `gp_validator.get_all_leak_positions(min_probability=0.75, min_distance=5.0)`
- Résolution de grille adaptative : 150×150 si < 10 mesures, 100×100 sinon
- Score combiné : `0.7 × concentration_normalisée + 0.3 × confiance` avec pénalité d'incertitude relative

### Calcul de la Confiance

**Confiance du Student :**
- Basée sur la perte RL moyenne des 10 dernières itérations
- Calculée à chaque étape d'apprentissage
- Utilisée pour ajuster les poids Teacher/Student

**Probabilité GP (Validateur) :**
- Probabilité retournée par le Validateur GP (0.0 à 1.0)
- Calculée comme score combiné : `0.7 × concentration_normalisée + 0.3 × confiance`
- Utilisée pour décider si l'estimation GP est fiable (seuils : 60% Phase 1, 30% Phase 2, 25% Phase 3)
- Influence les poids dans le mélange de directions selon la phase et la confiance
- Seuil adaptatif selon le nombre de mesures : réduit de 0.15 si < 10 mesures

---

## Conclusion

- **Mode Simple** : Mode basique, stable, rapide, pour tests et démonstrations simples. Pas d'IA, navigation directe basée sur gradient et direction.

- **Mode Teacher** : Mode expert, stable, rapide, optimal pour détection unique avec performance immédiate. Utilise uniquement le Teacher (GP) avec Validateur GP pour estimation probabiliste.

- **Mode Teacher-Student** : Mode adaptatif, optimal à long terme, meilleur pour détections multiples et optimisation continue. Combine Teacher (guidance stratégique) et Student (navigation tactique) avec stratégie adaptative.

Les **Mode Teacher** et **Mode Teacher-Student** sont maintenant **équivalents en performance initiale**, avec **Mode Teacher-Student** ayant un potentiel d'amélioration supplémentaire grâce à l'apprentissage du Student. Le choix dépend de vos besoins spécifiques : stabilité immédiate vs adaptation à long terme.
