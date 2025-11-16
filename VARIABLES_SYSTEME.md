# 📋 LISTE COMPLÈTE DES VARIABLES DU SYSTÈME HIGHLIGHT+

## 🌪️ **VARIABLES DU PANACHE DE MÉTHANE**

### **1. `leak_x` / `leak_y` (Position de la fuite)**
- **Type** : Float (mètres)
- **Défaut** : `50.0, 50.0`
- **Plage** : `0.0 - 100.0`
- **Signification** : Position (x, y) de la source de fuite de méthane dans le monde virtuel
- **Impact** :
  - ✅ Centre du panache de concentration
  - ✅ Plus le drone s'en approche, plus la concentration augmente
  - ✅ Influence la difficulté de détection (proximité du point de départ)
- **Recommandation** : Tester différentes positions pour valider la robustesse

---

### **2. `leak_intensity` (Intensité de la fuite)**
- **Type** : Float (kg/s)
- **Défaut** : `0.3`
- **Plage** : `0.01 - 1.0`
- **Signification** : Débit de fuite de méthane (Q dans la formule du panache)
- **Impact** :
  - ✅ **CRITIQUE** : Détermine l'amplitude maximale de la concentration
  - ✅ Plus élevé → Panache plus intense → Détection plus facile
  - ✅ Formule : `C ∝ Q / (σ_x σ_y u)` où Q = leak_intensity
  - ⚠️ Valeurs faibles (< 0.2) rendent la détection très difficile
- **Recommandation** :
  - Micro-fuites réelles : `0.1 - 0.3` kg/s
  - Tests : `0.3 - 0.5` kg/s
  - Démonstration : `0.5 - 1.0` kg/s

---

### **3. `wind_speed` (Vitesse du vent)**
- **Type** : Float (m/s)
- **Défaut** : `2.0`
- **Plage** : `0.0 - 10.0`
- **Signification** : Vitesse du vent qui transporte et disperse le panache
- **Impact** :
  - ✅ **CRITIQUE** : Influence la forme et l'étalement du panache
  - ✅ Vent fort → Panache plus étalé → Concentration maximale plus faible
  - ✅ Vent faible → Panache concentré → Concentration maximale plus élevée
  - ✅ Formule : `C ∝ 1 / u` où u = wind_speed
  - ⚠️ Vent > 5 m/s : Rend les détections difficiles (dispersion)
- **Recommandation** : `1.0 - 3.0` m/s pour conditions réalistes

---

### **4. `wind_direction` (Direction du vent)**
- **Type** : Float (degrés)
- **Défaut** : `45.0`
- **Plage** : `0.0 - 360.0`
- **Signification** : Direction du vent (0° = Nord, 90° = Est, etc.)
- **Impact** :
  - ✅ Détermine la direction d'étalement du panache
  - ✅ Influence la stratégie d'exploration (remonter le panache)
  - ✅ Panache s'étire dans la direction du vent
- **Recommandation** : Tester différentes directions pour robustesse

---

### **5. `sigma_x` / `sigma_y` (Paramètres de diffusion)**
- **Type** : Float (mètres)
- **Défaut** : `5.0, 3.0`
- **Plage** : `1.0 - 20.0`
- **Signification** : Écarts-types de la diffusion gaussienne (σ_x, σ_y)
- **Impact** :
  - ✅ **CRITIQUE** : Déterminent la largeur du panache
  - ✅ Valeurs élevées → Panache large et dilué
  - ✅ Valeurs faibles → Panache étroit et concentré
  - ✅ Formule : `C ∝ exp(-(x²/(2σ_x²) + y²/(2σ_y²)))`
- **Recommandation** : `3.0 - 8.0` m selon les conditions atmosphériques

---

### **6. `decay_rate` (Taux de décroissance)**
- **Type** : Float (s⁻¹)
- **Défaut** : `0.01`
- **Signification** : Taux de dégradation/dispersion temporelle du méthane
- **Impact** :
  - ✅ Affaiblit progressivement la concentration avec le temps
  - ✅ Formule : `C(t) = C(0) * exp(-decay_rate * t)`
- **Recommandation** : `0.005 - 0.02` s⁻¹

---

## 📡 **VARIABLES DU CAPTEUR TDLAS**

### **7. `detection_threshold` (Seuil de détection)**
- **Type** : Float (kg/m³)
- **Défaut** : `0.05`
- **Plage** : `0.001 - 1.0`
- **Signification** : Concentration minimale pour déclencher une détection
- **Impact** :
  - ✅ **CRITIQUE** : Détermine si une mesure est considérée comme détection
  - ✅ Valeur trop élevée → Aucune détection même avec panache réel
  - ✅ Valeur trop faible → Fausses détections (bruit)
  - ✅ Formule : `detected = measured_conc > detection_threshold`
- **Recommandation** :
  - Micro-fuites : `0.01 - 0.02` kg/m³
  - Tests standards : `0.02 - 0.05` kg/m³
  - Démonstration : `0.05 - 0.1` kg/m³

---

### **8. `noise_level` (Niveau de bruit)**
- **Type** : Float (σ)
- **Défaut** : `0.1`
- **Plage** : `0.01 - 1.0`
- **Signification** : Écart-type du bruit total du capteur
- **Impact** :
  - ✅ **CRITIQUE** : Ajoute du bruit à chaque mesure
  - ✅ Bruit élevé → Masque les faibles concentrations
  - ✅ Formule : `measured_conc = true_conc + N(0, noise_level²)`
  - ⚠️ Bruit > 0.1 peut masquer des détections réelles
- **Recommandation** :
  - Capteur de qualité : `0.02 - 0.05`
  - Capteur standard : `0.05 - 0.1`
  - Conditions difficiles : `0.1 - 0.2`

---

### **9. `range_max` / `range_min` (Portée du capteur)**
- **Type** : Float (mètres)
- **Défaut** : `100.0, 1.0`
- **Signification** : Distance minimale et maximale de mesure
- **Impact** :
  - ✅ Hors portée → Mesure retourne 0.0
  - ✅ Influencé par l'altitude (distance au sol)
- **Recommandation** : `50.0 - 200.0` m selon l'altitude

---

### **10. `atmospheric_noise` (Bruit atmosphérique)**
- **Type** : Float
- **Défaut** : `0.05`
- **Signification** : Bruit dû aux perturbations atmosphériques
- **Impact** :
  - ✅ Augmente avec la distance/altitude
  - ✅ Formule : `atmospheric_noise * sqrt(distance/10)`
- **Recommandation** : `0.02 - 0.1`

---

### **11. `electronic_noise` (Bruit électronique)**
- **Type** : Float
- **Défaut** : `0.02`
- **Signification** : Bruit intrinsèque de l'électronique du capteur
- **Impact** :
  - ✅ Constant, indépendant de la distance
  - ✅ Limite de précision du capteur
- **Recommandation** : `0.01 - 0.05`

---

## 🚁 **VARIABLES DU DRONE**

### **12. `initial_x` / `initial_y` (Position initiale)**
- **Type** : Float (mètres)
- **Défaut** : `10.0, 10.0`
- **Plage** : `0.0 - 100.0`
- **Signification** : Point de départ du drone
- **Impact** :
  - ✅ Influence le temps pour atteindre la zone de fuite
  - ✅ Distance initiale → Difficulté de la mission
- **Recommandation** : Tester différentes positions pour robustesse

---

### **13. `initial_altitude` (Altitude initiale)**
- **Type** : Float (mètres)
- **Défaut** : `5.0`
- **Plage** : `1.0 - 50.0`
- **Signification** : Altitude de départ du drone
- **Impact** :
  - ✅ **CRITIQUE** : Influence le SNR selon `I_s ∝ ρ / h²`
  - ✅ Plus bas → Meilleur signal → Meilleure détection
  - ✅ Plus haut → Signal faible → Détection difficile
- **Recommandation** : `3.0 - 10.0` m pour bon compromis

---

### **14. `max_speed` (Vitesse maximale)**
- **Type** : Float (m/s)
- **Défaut** : `5.0`
- **Plage** : `1.0 - 20.0`
- **Signification** : Vitesse maximale de déplacement
- **Impact** :
  - ✅ Plus rapide → Exploration plus rapide mais moins précise
  - ✅ Influence la consommation énergétique : `P ∝ v_air³`
  - ✅ Vitesse élevée → Plus d'énergie consommée
- **Recommandation** : `3.0 - 8.0` m/s selon le compromis vitesse/énergie

---

### **15. `max_altitude` / `min_altitude` (Limites d'altitude)**
- **Type** : Float (mètres)
- **Défaut** : `20.0, 2.0`
- **Signification** : Plage d'altitude autorisée
- **Impact** :
  - ✅ **CRITIQUE** : Contraint les mesures selon `I_s ∝ ρ / h²`
  - ✅ Altitude élevée → Signal faible → Détections difficiles
  - ✅ Altitude basse → Signal fort mais zone couverte réduite
- **Recommandation** :
  - Détection : `5.0 - 15.0` m
  - Exploration : `10.0 - 25.0` m

---

## 🧠 **VARIABLES DE L'IA (Teacher)**

### **16. `teacher_exploration` (Paramètre d'exploration)**
- **Type** : Float (β)
- **Défaut** : `2.5`
- **Plage** : `0.1 - 10.0`
- **Signification** : Paramètre β de la fonction d'acquisition UCB
- **Impact** :
  - ✅ **CRITIQUE** : Équilibre exploration/exploitation du Teacher
  - ✅ Valeur élevée → Exploration plus agressive
  - ✅ Valeur faible → Exploitation des zones connues
  - ✅ Formule UCB : `acquisition = mean + β * std`
- **Recommandation** :
  - Début de mission : `3.0 - 5.0`
  - Exploration approfondie : `2.0 - 3.0`
  - Exploitation : `1.0 - 2.0`

---

### **17. `kernel_length_scale` (Échelle du noyau GP)**
- **Type** : Float (mètres)
- **Défaut** : `10.0`
- **Signification** : Distance caractéristique du Processus Gaussien
- **Impact** :
  - ✅ Détermine la corrélation spatiale des mesures
  - ✅ Valeur élevée → Corrélation sur grande distance
  - ✅ Valeur faible → Corrélation locale seulement
- **Recommandation** : `5.0 - 15.0` m selon la taille du panache

---

### **18. `acquisition_function` (Fonction d'acquisition)**
- **Type** : String
- **Défaut** : `"UCB"`
- **Options** : `"UCB"`, `"EI"`, `"PI"`
- **Signification** : Stratégie de sélection du prochain point
- **Impact** :
  - ✅ UCB : Exploration/exploitation équilibrée
  - ✅ EI : Focus sur amélioration attendue
  - ✅ PI : Focus sur probabilité d'amélioration
- **Recommandation** : `"UCB"` pour missions de détection

---

## 🎓 **VARIABLES DE L'IA (Student)**

### **19. `student_learning_rate` (Taux d'apprentissage)**
- **Type** : Float
- **Défaut** : `1e-3`
- **Plage** : `1e-5 - 1e-1`
- **Signification** : Vitesse d'apprentissage du réseau de neurones
- **Impact** :
  - ✅ **CRITIQUE** : Vitesse de convergence de l'apprentissage
  - ✅ Valeur élevée → Apprentissage rapide mais instable
  - ✅ Valeur faible → Apprentissage lent mais stable
- **Recommandation** :
  - Début : `1e-3` (0.001)
  - Fine-tuning : `1e-4` (0.0001)

---

### **20. `student_lambda_kl` (Poids de distillation)**
- **Type** : Float (λ)
- **Défaut** : `0.2`
- **Plage** : `0.01 - 1.0`
- **Signification** : Poids de la perte de distillation KL
- **Impact** :
  - ✅ **CRITIQUE** : Équilibre entre RL et imitation du Teacher
  - ✅ Formule : `Loss = Loss_RL + λ * D_KL(π_Teacher || π_Student)`
  - ✅ Valeur élevée → Suit davantage le Teacher
  - ✅ Valeur faible → Apprentissage RL indépendant
- **Recommandation** : `0.1 - 0.3` pour bon équilibre

---

### **21. `epsilon_start` / `epsilon_end` (Exploration epsilon)**
- **Type** : Float
- **Défaut** : `1.0, 0.01`
- **Signification** : Exploration aléatoire (ε-greedy)
- **Impact** :
  - ✅ `epsilon_start` : Exploration initiale (100%)
  - ✅ `epsilon_end` : Exploration finale (1%)
  - ✅ Décroît linéairement pendant l'apprentissage
- **Recommandation** : Défauts sont bons

---

### **22. `epsilon_decay` (Vitesse de décroissance epsilon)**
- **Type** : Integer (étapes)
- **Défaut** : `10000`
- **Signification** : Nombre d'étapes pour passer de start à end
- **Impact** :
  - ✅ Valeur élevée → Exploration longue
  - ✅ Valeur faible → Exploitation rapide
- **Recommandation** : `5000 - 20000` selon la durée de mission

---

### **23. `batch_size` (Taille du batch)**
- **Type** : Integer
- **Défaut** : `64`
- **Signification** : Nombre d'expériences par étape d'apprentissage
- **Impact** :
  - ✅ Plus grand → Apprentissage plus stable mais plus lent
  - ✅ Plus petit → Apprentissage plus rapide mais plus bruité
- **Recommandation** : `32 - 128`

---

### **24. `buffer_size` (Taille du buffer)**
- **Type** : Integer
- **Défaut** : `10000`
- **Signification** : Nombre d'expériences stockées
- **Impact** :
  - ✅ Plus grand → Meilleure diversité mais plus de mémoire
  - ✅ Plus petit → Moins de mémoire mais risque de sur-apprentissage
- **Recommandation** : `5000 - 50000` selon la RAM disponible

---

### **25. `learning_starts` (Début de l'apprentissage)**
- **Type** : Integer (étapes)
- **Défaut** : `1000`
- **Signification** : Nombre d'expériences avant de commencer l'apprentissage
- **Impact** :
  - ✅ Permet de remplir le buffer avant d'apprendre
  - ✅ Évite l'apprentissage sur trop peu de données
- **Recommandation** : `500 - 2000`

---

## ⚙️ **VARIABLES DE L'ENVIRONNEMENT**

### **26. `max_steps` (Nombre d'étapes)**
- **Type** : Integer
- **Défaut** : `500`
- **Plage** : `100 - 2000`
- **Signification** : Durée maximale de la simulation
- **Impact** :
  - ✅ Plus long → Plus de temps pour détecter
  - ✅ Plus court → Simulation plus rapide
  - ⚠️ Doit être suffisant pour permettre l'exploration
- **Recommandation** :
  - Tests rapides : `200 - 500`
  - Simulations complètes : `500 - 1000`
  - Entraînement : `1000 - 2000`

---

### **27. `world_size` (Taille du monde)**
- **Type** : Tuple (mètres)
- **Défaut** : `(100.0, 100.0)`
- **Signification** : Dimensions de la zone de recherche
- **Impact** :
  - ✅ Plus grand → Mission plus longue
  - ✅ Plus petit → Mission plus rapide
- **Recommandation** : `(100, 100)` standard

---

### **28. `time_step` (Pas de temps)**
- **Type** : Float (secondes)
- **Défaut** : `0.1`
- **Signification** : Intervalle entre deux étapes
- **Impact** :
  - ✅ Plus petit → Simulation plus précise mais plus lente
  - ✅ Plus grand → Simulation plus rapide mais moins précise
- **Recommandation** : `0.05 - 0.2` s

---

## ⚡ **VARIABLES ÉNERGÉTIQUES**

### **29. `base_power` (Puissance de base)**
- **Type** : Float (Watts)
- **Défaut** : `100.0`
- **Signification** : Puissance consommée même au repos
- **Impact** :
  - ✅ Coût énergétique minimal
  - ✅ Formule : `P_total = base_power + P_mouvement`
- **Recommandation** : `50 - 200` W selon le drone

---

### **30. `speed_coefficient` (Coefficient vitesse)**
- **Type** : Float (W/(m/s)² pour normalisation)
- **Défaut** : `50.0`
- **Signification** : Coefficient pour `P ∝ v_air³`
- **Impact** :
  - ✅ **CRITIQUE** : Détermine le coût énergétique du mouvement
  - ✅ Formule : `P = c₁ * v_air³` où c₁ = speed_coefficient / max_speed²
- **Recommandation** : Ajuster selon les caractéristiques du drone

---

### **31. `altitude_coefficient` (Coefficient altitude)**
- **Type** : Float (W/m)
- **Défaut** : `25.0`
- **Signification** : Coefficient pour `P ∝ |Δh/Δt|`
- **Impact** :
  - ✅ Coût énergétique des changements d'altitude
  - ✅ Formule : `P = c₂ * |Δh/Δt|` où c₂ = altitude_coefficient
- **Recommandation** : `10 - 50` W/m

---

## 🎯 **VARIABLES DE RÉCOMPENSE**

### **32. `detection_bonus` (Bonus détection)**
- **Type** : Float
- **Défaut** : `10.0`
- **Signification** : Bonus pour chaque détection
- **Impact** :
  - ✅ Encourage les détections
  - ✅ Formule : `reward += detection_bonus if detected`
- **Recommandation** : `5.0 - 20.0`

---

### **33. `energy_penalty` (Pénalité énergétique)**
- **Type** : Float (négatif)
- **Défaut** : `-0.1`
- **Signification** : Pénalité par unité d'énergie consommée
- **Impact** :
  - ✅ **CRITIQUE** : Coefficient β dans `R = α*ΔI - β*E`
  - ✅ Encourage l'efficacité énergétique
  - ✅ Formule : `reward -= β * energy_cost`
- **Recommandation** : `-0.05` à `-0.2`

---

### **34. `exploration_bonus` (Bonus exploration)**
- **Type** : Float
- **Défaut** : `1.0`
- **Signification** : Bonus basé sur le gradient (exploration)
- **Impact** :
  - ✅ Encourage l'exploration des zones à fort gradient
  - ✅ Formule : `reward += exploration_bonus * ||gradient||`
- **Recommandation** : `0.5 - 2.0`

---

### **35. `boundary_penalty` (Pénalité limites)**
- **Type** : Float (négatif)
- **Défaut** : `-5.0`
- **Signification** : Pénalité pour sortir des limites
- **Impact** :
  - ✅ Décourage les sorties de zone
- **Recommandation** : `-5.0` à `-10.0`

---

## 📊 **TABLEAU RÉCAPITULATIF DES IMPACTS CRITIQUES**

| Variable | Impact Détection | Impact Énergie | Impact Performance |
|----------|-----------------|----------------|-------------------|
| `leak_intensity` | ⭐⭐⭐ CRITIQUE | - | ⭐⭐ |
| `detection_threshold` | ⭐⭐⭐ CRITIQUE | - | ⭐⭐⭐ |
| `noise_level` | ⭐⭐⭐ CRITIQUE | - | ⭐⭐ |
| `wind_speed` | ⭐⭐ | - | ⭐ |
| `max_altitude` | ⭐⭐ | ⭐ | ⭐ |
| `teacher_exploration` | ⭐⭐ | ⭐ | ⭐⭐ |
| `student_learning_rate` | - | - | ⭐⭐ |
| `energy_penalty` | - | ⭐⭐ | ⭐ |

---

## 🎯 **RECOMMANDATIONS PAR OBJECTIF**

### **✅ Maximiser les détections :**
1. Réduire `detection_threshold` à `0.01`
2. Réduire `noise_level` à `0.05`
3. Augmenter `leak_intensity` à `0.5`
4. Réduire `max_altitude` à `10.0`
5. Augmenter `teacher_exploration` à `3.0`

### **⚡ Optimiser l'énergie :**
1. Réduire `max_speed` à `3.0`
2. Augmenter `energy_penalty` à `-0.2`
3. Réduire `base_power` si possible
4. Minimiser les changements d'altitude

### **🎯 Équilibre détection/énergie :**
1. `detection_threshold` : `0.02`
2. `leak_intensity` : `0.3`
3. `max_altitude` : `15.0`
4. `teacher_exploration` : `2.5`
5. `energy_penalty` : `-0.1`

---

## 📝 **NOTES IMPORTANTES**

- **Toutes les valeurs par défaut** sont configurées pour un fonctionnement équilibré
- **Les variables marquées CRITIQUE** ont un impact majeur sur les résultats
- **Ajuster progressivement** : Changer une variable à la fois pour comprendre son impact
- **Logs détaillés** : Utiliser les logs périodiques pour diagnostiquer les problèmes

---

## 🔧 **OUTILS DE DIAGNOSTIC**

Le système génère automatiquement :
- 📊 Métriques de performance (taux de détection, énergie, etc.)
- 📍 Logs périodiques des concentrations (toutes les 50 étapes)
- 🎯 Position de chaque détection
- 📈 Statistiques du Teacher (nombre d'observations)

**Utilisez ces informations pour ajuster les variables selon vos résultats !**




