"""
Détecteur amélioré avec validation robuste pour HIGHLIGHT+
Intègre filtrage intelligent, validation multi-critères et estimation de confiance
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.spatial.distance import pdist, cdist


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
                 min_distance_for_detection: float = 50.0):
        """
        Args:
            true_leak_position: Position réelle de la fuite
            detection_threshold: Seuil de détection (concentration)
            confidence_threshold: Seuil de confiance minimum
            min_distance_for_detection: Distance minimale pour accepter une détection
        """
        self.true_leak_position = np.array(true_leak_position)
        self.detection_threshold = detection_threshold
        self.confidence_threshold = confidence_threshold
        self.min_distance_for_detection = min_distance_for_detection
        
        self.detections: List[DetectionEvent] = []
        self.concentration_history: List[float] = []
        self.position_history: List[np.ndarray] = []
        self.convergence_step: Optional[int] = None
        
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
        
        Returns:
            (position_estimée, confiance_estimation)
        """
        if not self.detections:
            return None, 0.0
        
        valid_detections = [d for d in self.detections if d.is_valid]
        if not valid_detections:
            return None, 0.0
        
        # Si une seule détection, utiliser celle-ci
        if len(valid_detections) == 1:
            det = valid_detections[0]
            confidence = det.confidence
            return det.position[:2].copy(), confidence
        
        # Filtrage des outliers
        positions = np.array([d.position[:2] for d in valid_detections])
        concentrations = np.array([d.concentration for d in valid_detections])
        confidences = np.array([d.confidence for d in valid_detections])
        
        # Calcul de la distance médiane
        if len(positions) > 1:
            pairwise_distances = pdist(positions)
            median_distance = np.median(pairwise_distances)
            
            # Filtrer les outliers (trop éloignés des autres)
            valid_indices = []
            for i, pos in enumerate(positions):
                distances_to_others = np.array([
                    np.linalg.norm(pos - positions[j])
                    for j in range(len(positions)) if i != j
                ])
                if len(distances_to_others) > 0:
                    min_distance = np.min(distances_to_others)
                    if min_distance < 2 * median_distance:
                        valid_indices.append(i)
            
            if len(valid_indices) >= 2:
                positions = positions[valid_indices]
                concentrations = concentrations[valid_indices]
                confidences = confidences[valid_indices]
        
        # Estimation par moyenne pondérée
        # Poids = concentration * confidence
        weights = concentrations * confidences
        if np.sum(weights) > 1e-6:
            weights = weights / np.sum(weights)
            estimated_position = np.average(positions, axis=0, weights=weights)
            
            # Confiance globale = moyenne pondérée des confiances
            global_confidence = np.average(confidences, weights=weights)
            
            return estimated_position, global_confidence
        
        # Fallback: moyenne simple
        estimated_position = np.mean(positions, axis=0)
        global_confidence = np.mean(confidences)
        
        return estimated_position, global_confidence
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de détection"""
        if not self.detections:
            return {
                'n_detections': 0,
                'n_valid_detections': 0,
                'avg_confidence': 0.0,
                'avg_distance': 0.0,
                'convergence_step': None
            }
        
        valid_detections = [d for d in self.detections if d.is_valid]
        
        return {
            'n_detections': len(self.detections),
            'n_valid_detections': len(valid_detections),
            'avg_confidence': np.mean([d.confidence for d in valid_detections]) if valid_detections else 0.0,
            'avg_distance': np.mean([d.distance_to_source for d in valid_detections]) if valid_detections else 0.0,
            'min_distance': min([d.distance_to_source for d in valid_detections]) if valid_detections else float('inf'),
            'convergence_step': self.convergence_step,
            'first_detection_step': self.detections[0].step if self.detections else None
        }




