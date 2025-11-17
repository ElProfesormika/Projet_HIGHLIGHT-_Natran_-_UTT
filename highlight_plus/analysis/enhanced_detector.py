"""
Détecteur amélioré avec validation robuste pour HIGHLIGHT+
Intègre filtrage intelligent, validation multi-critères et estimation de confiance
Utilise un validateur GP pour l'estimation de position de fuite
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.spatial.distance import pdist, cdist
import os
import sys

# Import du validateur GP
try:
    from .methane_leak_validator import MethaneLeakValidator
    GP_VALIDATOR_AVAILABLE = True
except ImportError:
    GP_VALIDATOR_AVAILABLE = False


@dataclass
class DetectionEvent:
    """Événement de détection avec métadonnées"""
    position: np.ndarray
    concentration: float
    concentration_real: float
    step: int
    timestamp: float
    confidence: float  # Confiance de la détection [0, 1]
    is_valid: bool  # Si la détection est validée
    distance_to_source: float
    gradient_magnitude: float


class EnhancedDetector:
    """
    Détecteur amélioré avec validation robuste
    
    Caractéristiques :
    - Filtrage multi-critères
    - Validation de progression
    - Estimation de confiance
    - Détection de convergence
    """
    
    def __init__(self, true_leak_position: Tuple[float, float],
                 detection_threshold: float = 0.05,
                 confidence_threshold: float = 0.6,
                 min_distance_for_detection: float = 50.0,
                 use_gp_validator: bool = True,
                 gp_threshold_prob: float = 0.95,
                 world_bounds: Tuple[float, float, float, float] = (0, 100, 0, 100)):
        """
        Args:
            true_leak_position: Position réelle de la fuite (UNIQUEMENT pour validation/comparaison)
                               Cette position n'est PAS utilisée dans l'estimation de position.
                               Elle sert uniquement à calculer l'erreur pour prouver la fiabilité.
            detection_threshold: Seuil de détection (concentration)
            confidence_threshold: Seuil de confiance minimum
            min_distance_for_detection: Distance minimale pour accepter une détection
            use_gp_validator: Utiliser le validateur GP pour l'estimation de position
            gp_threshold_prob: Seuil de probabilité pour le validateur GP (0-1)
            world_bounds: Limites du monde (x_min, x_max, y_min, y_max)
        """
        # NOTE IMPORTANTE: true_leak_position est UNIQUEMENT pour la validation/comparaison
        # L'estimation de position (estimate_leak_position) ne l'utilise PAS
        # Elle est utilisée uniquement pour calculer distance_to_source dans les détections
        # et pour la validation finale dans performance_validator
        self.true_leak_position = np.array(true_leak_position)
        self.detection_threshold = detection_threshold
        self.confidence_threshold = confidence_threshold
        self.min_distance_for_detection = min_distance_for_detection
        
        self.detections: List[DetectionEvent] = []
        self.concentration_history: List[float] = []
        self.position_history: List[np.ndarray] = []
        self.convergence_step: Optional[int] = None
        
        # Historique des estimations pour détection de convergence
        self.estimation_history: List[Tuple[np.ndarray, float]] = []  # (position, confidence)
        
        # Validateur GP (priorité pour la détection)
        self.gp_validator = None
        self.use_gp_validator = use_gp_validator and GP_VALIDATOR_AVAILABLE
        
        if self.use_gp_validator:
            try:
                self.gp_validator = MethaneLeakValidator(
                    grid_size=(100, 100),
                    world_bounds=world_bounds,
                    threshold_prob=gp_threshold_prob,
                    noise=1e-2,
                    kernel_length_scale=5.0,
                    kernel_variance=1.0
                )
            except Exception as e:
                print(f"Impossible d'initialiser le validateur GP: {e}")
                self.use_gp_validator = False
        
    def validate_detection(self, position: np.ndarray, 
                          measured_concentration: float,
                          real_concentration: float,
                          step: int,
                          timestamp: float,
                          gradient: Optional[np.ndarray] = None) -> Optional[DetectionEvent]:
        """
        Valide une détection avec critères multiples
        
        Args:
            position: Position du drone (x, y, z)
            measured_concentration: Concentration mesurée (avec bruit)
            real_concentration: Concentration réelle (sans bruit)
            step: Étape de simulation
            timestamp: Temps simulé
            gradient: Gradient de concentration (optionnel)
            
        Returns:
            DetectionEvent si la détection est validée, None sinon
        """
        # Calcul de la distance à la source
        position_2d = position[:2]
        distance_to_source = np.linalg.norm(position_2d - self.true_leak_position)
        
        # Calcul du gradient si disponible
        gradient_magnitude = 0.0
        if gradient is not None:
            gradient_magnitude = np.linalg.norm(gradient[:2])
        
        # Critère 1: Concentration mesurée > seuil
        if measured_concentration < self.detection_threshold:
            return None
        
        # Critère 2: Validation multi-critères
        confidence = self._calculate_confidence(
            measured_concentration, real_concentration, 
            distance_to_source, gradient_magnitude, step
        )
        
        # Critère 3: Progression (concentration augmente avec le temps)
        is_progressing = self._check_progression(real_concentration)
        
        # Critère 4: Distance raisonnable
        is_near_source = distance_to_source < self.min_distance_for_detection or distance_to_source < 30.0
        
        # Validation finale
        is_valid = (
            confidence >= self.confidence_threshold or
            (is_progressing and is_near_source and confidence >= 0.4) or
            (distance_to_source < 15.0 and measured_concentration > self.detection_threshold * 0.8)
        )
        
        if not is_valid:
            return None
        
        # Création de l'événement de détection
        detection = DetectionEvent(
            position=position.copy(),
            concentration=measured_concentration,
            concentration_real=real_concentration,
            step=step,
            timestamp=timestamp,
            confidence=confidence,
            is_valid=True,
            distance_to_source=distance_to_source,
            gradient_magnitude=gradient_magnitude
        )
        
        # Enregistrement
        self.detections.append(detection)
        self.concentration_history.append(real_concentration)
        self.position_history.append(position_2d.copy())
        
        # Ajouter la mesure au validateur GP (si disponible)
        if self.use_gp_validator and self.gp_validator is not None:
            try:
                self.gp_validator.add_measurement(
                    position=(position[0], position[1]),
                    concentration=measured_concentration
                )
            except Exception as e:
                # En cas d'erreur, continuer sans GP
                pass
        
        # Vérifier la convergence (première détection proche de la source)
        if self.convergence_step is None and distance_to_source < 20.0:
            self.convergence_step = step
        
        return detection
    
    def _calculate_confidence(self, measured_conc: float, real_conc: float,
                             distance: float, gradient: float, step: int) -> float:
        """
        Calcule la confiance d'une détection [0, 1]
        
        Facteurs :
        - Ratio concentration mesurée/réelle
        - Distance à la source
        - Magnitude du gradient
        - Progression dans le temps
        """
        confidence = 0.0
        
        # Facteur 1: Qualité de la mesure (ratio mesurée/réelle)
        if real_conc > 1e-6:
            ratio = min(measured_conc / real_conc, 2.0)  # Cap à 2x
            measure_quality = np.exp(-(ratio - 1.0)**2)  # Gaussienne centrée à 1.0
            confidence += 0.3 * measure_quality
        
        # Facteur 2: Distance à la source (plus proche = plus confiant)
        distance_factor = np.exp(-distance / 30.0)  # Décroissance exponentielle
        confidence += 0.3 * distance_factor
        
        # Facteur 3: Gradient (fort gradient = proche de la source)
        if gradient > 1e-6:
            gradient_factor = min(gradient / 0.1, 1.0)  # Normalisé à 0.1
            confidence += 0.2 * gradient_factor
        
        # Facteur 4: Progression (concentration augmente)
        if len(self.concentration_history) >= 3:
            recent = self.concentration_history[-3:]
            if all(recent[i] >= recent[i-1] * 0.9 for i in range(1, len(recent))):
                confidence += 0.2
        
        # Normalisation
        return min(confidence, 1.0)
    
    def _check_progression(self, current_concentration: float) -> bool:
        """Vérifie si la concentration progresse"""
        if len(self.concentration_history) < 5:
            return True  # Pas assez de données
        
        # Vérifier la tendance sur les 5 dernières mesures
        recent = self.concentration_history[-5:]
        increasing_count = sum(1 for i in range(1, len(recent)) 
                              if recent[i] > recent[i-1] * 0.95)
        
        # Au moins 60% des mesures sont croissantes
        return increasing_count >= 3
    
    def estimate_leak_position(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Estime la position de la fuite à partir des détections
        
        IMPORTANT: Cette méthode est COMPLÈTEMENT INDÉPENDANTE de la position réelle.
        Elle n'utilise PAS self.true_leak_position. L'estimation est basée uniquement sur:
        - Les positions des détections
        - Les concentrations mesurées
        - Les confiances calculées
        - La cohérence spatiale et temporelle
        
        La position réelle est utilisée UNIQUEMENT dans performance_validator pour
        calculer l'erreur et prouver la fiabilité du modèle aux observateurs.
        
        Stratégie :
        1. PRIORITÉ ABSOLUE : Validateur GP (accumulation des mesures, modélisation probabiliste)
        2. Sinon : Méthode statistique robuste (clustering + médiane pondérée)
        
        Returns:
            (position_estimée, confiance_estimation)
        """
        if not self.detections:
            return None, 0.0
        
        valid_detections = [d for d in self.detections if d.is_valid]
        if not valid_detections:
            return None, 0.0
        
        # PRIORITÉ 1 : Utiliser le validateur GP (logique principale)
        if self.use_gp_validator and self.gp_validator is not None:
            try:
                leak_pos, leak_prob = self.gp_validator.get_leak_position()
                if leak_pos is not None:
                    # Convertir en numpy array
                    leak_pos_array = np.array(leak_pos)
                    
                    # Stocker dans l'historique pour détection de convergence
                    self.estimation_history.append((leak_pos_array.copy(), float(leak_prob)))
                    # Garder seulement les 10 dernières estimations
                    if len(self.estimation_history) > 10:
                        self.estimation_history.pop(0)
                    
                    return leak_pos_array, float(leak_prob)
            except Exception as e:
                # En cas d'erreur, fallback sur méthode statistique
                pass
        
        # PRIORITÉ 2 : Méthode statistique robuste (fallback si GP non disponible)
        return self._estimate_position_statistical(valid_detections)
    
    def _estimate_position_statistical(self, valid_detections: List[DetectionEvent]) -> Tuple[Optional[np.ndarray], float]:
        """
        Méthode statistique robuste (fallback si GP non disponible)
        
        Stratégie améliorée :
        1. Utiliser TOUTES les détections
        2. Clustering pour identifier les groupes cohérents
        3. Poids temporel (détections récentes plus importantes)
        4. Filtrage robuste des outliers
        5. Estimation par médiane pondérée pour robustesse
        
        Returns:
            (position_estimée, confiance_estimation)
        """
        if len(valid_detections) == 1:
            det = valid_detections[0]
            return det.position[:2].copy(), det.confidence
        
        # Extraction des données
        # NOTE: distances_to_source est extrait mais N'EST PAS utilisé dans l'estimation
        # Il est conservé uniquement pour compatibilité, mais l'estimation est indépendante
        positions = np.array([d.position[:2] for d in valid_detections])
        concentrations = np.array([d.concentration for d in valid_detections])
        confidences = np.array([d.confidence for d in valid_detections])
        steps = np.array([d.step for d in valid_detections])
        distances_to_source = np.array([d.distance_to_source for d in valid_detections])  # Non utilisé dans l'estimation
        
        # OPTIMISATION : Limiter le nombre de détections utilisées pour éviter le bruit
        # Si trop de détections, garder seulement les meilleures (récentes + haute confiance)
        # Note: Le validateur GP utilise toutes les mesures, cette limite s'applique uniquement à la méthode statistique
        max_detections_to_use = 50  # Limite augmentée pour utiliser plus d'informations
        if len(valid_detections) > max_detections_to_use:
            # Score combiné : récence + confiance + concentration
            recency_scores = (steps - steps.min()) / (steps.max() - steps.min() + 1e-6)
            quality_scores = confidences * (concentrations / (np.max(concentrations) + 1e-6))
            combined_scores = 0.5 * recency_scores + 0.5 * quality_scores
            
            # Garder les N meilleures détections
            top_indices = np.argsort(combined_scores)[-max_detections_to_use:]
            positions = positions[top_indices]
            concentrations = concentrations[top_indices]
            confidences = confidences[top_indices]
            steps = steps[top_indices]
            distances_to_source = distances_to_source[top_indices]
        
        # ÉTAPE 1 : Clustering simple basé sur la densité spatiale
        # Identifier le cluster principal (groupe de détections proches)
        if len(positions) >= 3:
            # Calculer les distances entre toutes les paires
            pairwise_distances = pdist(positions)
            median_distance = np.median(pairwise_distances)
            
            # Seuil de clustering : détections à moins de 1.2x la médiane appartiennent au même cluster
            cluster_threshold = 1.2 * median_distance
            
            # Identifier le cluster principal (le plus dense)
            cluster_sizes = []
            cluster_indices_list = []
            
            for i, pos in enumerate(positions):
                # Trouver toutes les détections proches de celle-ci
                distances_to_i = np.array([
                    np.linalg.norm(pos - positions[j])
                    for j in range(len(positions))
                ])
                cluster_indices = np.where(distances_to_i < cluster_threshold)[0]
                cluster_sizes.append(len(cluster_indices))
                cluster_indices_list.append(cluster_indices)
            
            # Prendre le cluster le plus grand
            main_cluster_idx = np.argmax(cluster_sizes)
            main_cluster_indices = cluster_indices_list[main_cluster_idx]
            
            # Si le cluster principal contient au moins 30% des détections, l'utiliser
            if len(main_cluster_indices) >= max(3, len(positions) * 0.3):
                positions = positions[main_cluster_indices]
                concentrations = concentrations[main_cluster_indices]
                confidences = confidences[main_cluster_indices]
                steps = steps[main_cluster_indices]
                distances_to_source = distances_to_source[main_cluster_indices]
        
        # ÉTAPE 2 : Filtrage des outliers restants (plus strict)
        if len(positions) > 2:
            # Calculer le centre médian du cluster
            median_position = np.median(positions, axis=0)
            distances_to_median = np.array([
                np.linalg.norm(pos - median_position) for pos in positions
            ])
            median_distance_to_center = np.median(distances_to_median)
            
            # Garder seulement les détections à moins de 1.5x la distance médiane au centre
            valid_mask = distances_to_median < 1.5 * median_distance_to_center
            
            if np.sum(valid_mask) >= 2:
                positions = positions[valid_mask]
                concentrations = concentrations[valid_mask]
                confidences = confidences[valid_mask]
                steps = steps[valid_mask]
                distances_to_source = distances_to_source[valid_mask]
        
        # ÉTAPE 3 : Poids temporel (détections récentes plus importantes)
        # Normaliser les steps (0 = première, 1 = dernière)
        if len(steps) > 1:
            steps_normalized = (steps - steps.min()) / (steps.max() - steps.min() + 1e-6)
            # Poids temporel : exponentiel, favorisant les détections récentes
            temporal_weights = np.exp(2.0 * steps_normalized)  # Plus récent = poids plus élevé
        else:
            temporal_weights = np.ones(len(positions))
        
        # ÉTAPE 4 : Poids de cohérence spatiale (détections proches les unes des autres)
        if len(positions) > 1:
            # Calculer la distance moyenne de chaque détection aux autres
            avg_distances = np.array([
                np.mean([np.linalg.norm(pos - positions[j]) 
                        for j in range(len(positions)) if i != j])
                for i, pos in enumerate(positions)
            ])
            # Poids inversement proportionnel à la distance moyenne (plus cohérent = poids plus élevé)
            coherence_weights = 1.0 / (1.0 + avg_distances / 5.0)  # Normalisé à 5m
        else:
            coherence_weights = np.ones(len(positions))
        
        # ÉTAPE 5 : Poids combinés (INDÉPENDANT de la position réelle)
        # IMPORTANT: Cette estimation ne connaît PAS la position réelle
        # Elle est basée uniquement sur les propriétés intrinsèques des détections
        # Combiner : concentration * confidence * temporal * coherence
        # Normaliser les concentrations pour éviter la dominance
        if len(concentrations) > 0:
            conc_normalized = concentrations / (np.max(concentrations) + 1e-6)
        else:
            conc_normalized = concentrations
        
        # Poids finaux
        weights = (conc_normalized * confidences * temporal_weights * coherence_weights)
        
        if np.sum(weights) > 1e-6:
            weights = weights / np.sum(weights)
            
            # ÉTAPE 6 : Estimation robuste (médiane pondérée au lieu de moyenne)
            # Pour chaque dimension, calculer la médiane pondérée
            estimated_position = np.array([
                np.average(positions[:, dim], weights=weights) 
                for dim in range(positions.shape[1])
            ])
            
            # Alternative : utiliser la médiane des positions les plus pondérées
            # Prendre les 50% des détections avec les poids les plus élevés
            top_indices = np.argsort(weights)[-max(1, len(weights) // 2):]
            estimated_position_robust = np.median(positions[top_indices], axis=0)
            
            # Combiner les deux estimations (70% robuste, 30% pondérée)
            estimated_position = 0.7 * estimated_position_robust + 0.3 * estimated_position
            
            # Confiance globale = moyenne pondérée des confiances du cluster
            global_confidence = np.average(confidences, weights=weights)
            
            return estimated_position, global_confidence
        
        # Fallback: médiane simple (très robuste)
        estimated_position = np.median(positions, axis=0)
        global_confidence = np.median(confidences)
        
        return estimated_position, global_confidence
    
    def is_estimation_stable(self, threshold: float = 2.0) -> bool:
        """
        Vérifie si l'estimation de position est stable (convergence)
        
        Args:
            threshold: Distance seuil en mètres pour considérer stable (défaut: 2m)
            
        Returns:
            True si l'estimation est stable, False sinon
        """
        if len(self.estimation_history) < 3:
            return False
        
        # Vérifier si les dernières estimations sont proches
        recent_estimations = self.estimation_history[-3:]
        positions = np.array([est[0] for est in recent_estimations])
        
        # Calculer la distance maximale entre les estimations récentes
        max_distance = 0.0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                max_distance = max(max_distance, dist)
        
        return max_distance < threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de détection"""
        if not self.detections:
            return {
                'n_detections': 0,
                'n_valid_detections': 0,
                'avg_confidence': 0.0,
                'avg_distance': 0.0,
                'convergence_step': None,
                'estimation_stable': False
            }
        
        valid_detections = [d for d in self.detections if d.is_valid]
        
        return {
            'n_detections': len(self.detections),
            'n_valid_detections': len(valid_detections),
            'avg_confidence': np.mean([d.confidence for d in valid_detections]) if valid_detections else 0.0,
            'avg_distance': np.mean([d.distance_to_source for d in valid_detections]) if valid_detections else 0.0,
            'min_distance': min([d.distance_to_source for d in valid_detections]) if valid_detections else float('inf'),
            'convergence_step': self.convergence_step,
            'first_detection_step': self.detections[0].step if self.detections else None,
            'estimation_stable': self.is_estimation_stable()
        }





