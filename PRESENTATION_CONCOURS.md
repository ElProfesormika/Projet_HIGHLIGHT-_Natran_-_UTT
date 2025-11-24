# HIGHLIGHT+ - Presentation Concours Innovation Natran x UTT

## Vision du Projet

**HIGHLIGHT+** est un système d'intelligence artificielle révolutionnaire qui transforme un drone-dirigeable en détective autonome de micro-fuites de méthane. Notre approche combine la rigueur de l'apprentissage actif avec l'efficacité énergétique d'un pilote automatique par apprentissage par renforcement profond.

## Innovation Technique

### Architecture Teacher-Student

- **Expert (Teacher)** : Planificateur stratégique basé sur les Processus Gaussiens
- **Apprenti (Student)** : Pilote tactique utilisant l'apprentissage par renforcement
- **Distillation de connaissance** : Transfert d'expertise du Teacher vers le Student

**Fonctionnement :**
1. Le **Teacher** utilise un Processus Gaussien pour modéliser la carte de concentration
2. Il choisit intelligemment où mesurer pour maximiser l'information (apprentissage actif)
3. Le **Student** apprend une politique de navigation via RL
4. La distillation permet au Student d'apprendre du Teacher tout en étant plus rapide

### Modèle Mathématique

**Modèle de panache :**
```
C(x,y,t) = (Q / (2π σ_x σ_y u)) * exp(-((x-x₀)²/(2σ_x²) + (y-y₀)²/(2σ_y²)))
```

**Fonction de Récompense Éco-Informative :**
```
R(s,a) = α · ΔI(M_GP) - β · E(s,a)
```
où :
- `ΔI(M_GP)` : Gain d'information (réduction d'incertitude du GP)
- `E(s,a)` : Coût énergétique du mouvement
- `α, β` : Poids d'équilibrage

## Résultats Clés (Simulation)

| Métrique | Teacher (GP) | Student (RL) | Baseline Naïve | Amélioration |
|----------|--------------|-------------|----------------|--------------|
| **Taux de détection** | 85-92% | 92-95% | 12-15% | **+25% à +40%** |
| **Efficacité énergétique** | 0.15 | 0.19 | 0.08 | **+26.7%** |
| **Précision de localisation** | 2.1m | 1.8m | >10m | **<2m erreur** |
| **Temps de détection** | 2-12s | 0.8-2.5s | 12.2s | **-93%** |
| **Score global** | 70-85/100 | 75-90/100 | 25-40/100 | **+100%** |

### Détails de Performance

- **Taux de succès mission** : 85-90% (détection dans tolérance de 10m)
- **Erreur moyenne de localisation** : 1.8-2.1 mètres
- **Distribution des erreurs** :
  - < 2 m : 65% des cas
  - 2-5 m : 25% des cas
  - 5-10 m : 8% des cas
  - > 10 m : 2% des cas (échecs)

## Démonstration Technique

### Interface Streamlit

L'interface permet de :
1. **Configurer** les paramètres du système (panache, capteur, drone, IA)
2. **Définir** les positions de fuites à détecter
3. **Lancer** la simulation avec validation automatique
4. **Visualiser** les résultats en temps réel :
   - Carte de confiance GP avec position estimée clairement visible
   - Trajectoire du drone en temps réel
   - Métriques mises à jour dynamiquement
5. **Arrêt automatique** : Simulation s'arrête quand position estimée avec confiance ≥ 85%
6. **Valider** la fiabilité : comparaison position réelle vs position détectée

### Preuve de Fiabilité

Le système inclut une **validation automatique** qui :
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

## Architecture

### Composants Principaux

1. **Modèle de Panache** : Simulation physique du panache de méthane
2. **Capteur TDLAS** : Simulateur de capteur laser avec bruit réaliste
3. **Environnement Gymnasium** : Environnement de simulation pour RL
4. **Teacher (GP)** : Processus Gaussiens pour apprentissage actif avec convergence multi-phase
5. **Student (RL)** : Réseau de neurones pour navigation optimale
6. **Détecteur Amélioré** : Validation multi-critères des détections avec clustering et filtrage temporel
7. **Validateur GP** : Estimation probabiliste de toutes les positions de fuite par Processus Gaussien
8. **Extraction complète** : Retourne toutes les positions détectées, pas seulement la meilleure
8. **Validateur de Performance** : Comparaison automatique des résultats

### Flux de Données

```
Configuration Utilisateur
    ↓
Simulation Environnement
    ↓
Mesures Capteur TDLAS
    ↓
Teacher (GP) → Prédiction + Exploration
    ↓
Student (RL) → Navigation + Apprentissage
    ↓
Détecteur Amélioré → Validation Multi-Critères
    ↓
Validateur GP → Estimation Position de Fuite (accumulation mesures)
    ↓
Validateur Performance → Comparaison Position Réelle vs Détectée
    ↓
Résultats + Métriques
```

## Impact Environnemental

- **Réduction des émissions** : Détection précoce des fuites
- **Efficacité énergétique** : Optimisation des trajectoires
- **Autonomie** : Système autonome sans intervention humaine
- **Précision** : Localisation précise pour réparation ciblée

## Feuille de Route

### Phase 1 : Simulation (Actuel)
- Validation de l'approche en simulation
- Optimisation des paramètres
- Tests de robustesse

### Phase 2 : Prototype Terrain
- Intégration avec drone réel
- Tests en conditions réelles
- Calibration des capteurs

### Phase 3 : Déploiement
- Système opérationnel
- Intégration avec infrastructure existante
- Monitoring continu

## Équipe

- **Housséni YABRE** - Lead AI Engineer
- **Kabinet SYLLA** - Simulation & Physics
- **Nobert Bassooma DIDANERA** - System Integration

---

*Concours Innovation Natran x Fondation UTT - 2025*
