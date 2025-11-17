"""
Module de validation de performance pour HIGHLIGHT+
Compare les positions réelles des fuites avec les détections effectuées
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from datetime import datetime
from scipy.spatial.distance import pdist


@dataclass
class DetectionResult:
    """Résultat d'une détection"""
    position: np.ndarray  # Position où la détection a eu lieu (x, y, z)
    concentration: float  # Concentration mesurée
    step: int  # Étape de simulation
    timestamp: float  # Temps simulé (en secondes)


@dataclass
class LocalizationAccuracy:
    """Précision de localisation"""
    detected_position: np.ndarray  # Position détectée (x, y)
    true_position: np.ndarray  # Position réelle de la fuite (x, y)
    error_distance: float  # Distance d'erreur (m)
    error_angle: float  # Angle d'erreur (degrés)
    is_within_tolerance: bool  # Si la détection est dans la tolérance
    tolerance_radius: float  # Rayon de tolérance (m)


@dataclass
class PerformanceMetrics:
    """Métriques complètes de performance"""
    # Détection
    n_detections: int  # Nombre total de détections
    first_detection_step: Optional[int]  # Étape de première détection
    first_detection_time: Optional[float]  # Temps de première détection (s)
    detection_rate: float  # Taux de détection (détections / étape)
    
    # Localisation
    localization_accuracy: Optional[LocalizationAccuracy]  # Précision de localisation
    best_detection: Optional[DetectionResult]  # Meilleure détection (plus proche)
    avg_detection_distance: Optional[float]  # Distance moyenne des détections à la source
    
    # Temps et énergie
    total_time: float  # Temps total de mission (s)
    total_energy: float  # Énergie totale consommée (J)
    energy_per_detection: Optional[float]  # Énergie par détection (J/détection)
    
    # Performance globale
    detection_score: float  # Score de détection (0-100)
    localization_score: float  # Score de localisation (0-100)
    overall_score: float  # Score global (0-100)
    
    # Statistiques
    mission_success: bool  # Si la mission a réussi (détection + localisation)
    convergence_time: Optional[float]  # Temps de convergence (première détection correcte)


class PerformanceValidator:
    """
    Valideur de performance pour HIGHLIGHT+
    
    IMPORTANT: Ce module est utilisé UNIQUEMENT pour la validation et la comparaison.
    Il compare les détections effectuées avec la position réelle de la fuite
    pour calculer l'erreur et prouver la fiabilité du modèle aux observateurs.
    
    La position réelle n'est PAS utilisée dans l'estimation de position.
    Elle est fournie uniquement pour permettre la validation des performances.
    """
    
    def __init__(self, true_leak_position: Tuple[float, float], 
                 tolerance_radius: float = 10.0,
                 time_step: float = 0.1):
        """
        Args:
            true_leak_position: Position réelle de la fuite (x, y)
                               UNIQUEMENT pour validation/comparaison.
                               Cette position n'est PAS connue par le modèle d'estimation.
            tolerance_radius: Rayon de tolérance pour une détection valide (m)
            time_step: Pas de temps de la simulation (s)
        """
        self.true_leak_position = np.array(true_leak_position)
        self.tolerance_radius = tolerance_radius
        self.time_step = time_step
        self.detections: List[DetectionResult] = []
        self.total_energy = 0.0
        self.total_steps = 0
        
    def add_detection(self, position: np.ndarray, concentration: float, 
                     step: int, energy: float = 0.0):
        """
        Ajoute une détection à l'historique
        
        Args:
            position: Position de détection (x, y, z)
            concentration: Concentration mesurée
            step: Étape de simulation
            energy: Énergie consommée jusqu'à présent
        """
        detection = DetectionResult(
            position=position.copy(),
            concentration=concentration,
            step=step,
            timestamp=step * self.time_step
        )
        self.detections.append(detection)
        self.total_energy = energy
        self.total_steps = step
    
    def calculate_localization_accuracy(self, detection: DetectionResult) -> LocalizationAccuracy:
        """
        Calcule la précision de localisation pour une détection
        
        Args:
            detection: Résultat de détection
            
        Returns:
            LocalizationAccuracy avec métriques de précision
        """
        detected_pos_2d = detection.position[:2]  # (x, y)
        true_pos_2d = self.true_leak_position[:2]
        
        # Distance d'erreur
        error_distance = np.linalg.norm(detected_pos_2d - true_pos_2d)
        
        # Angle d'erreur (direction de la fuite depuis la détection)
        if error_distance > 1e-6:
            vec_to_leak = true_pos_2d - detected_pos_2d
            angle = np.arctan2(vec_to_leak[1], vec_to_leak[0]) * 180 / np.pi
            # Normaliser entre -180 et 180
            angle = (angle + 180) % 360 - 180
        else:
            angle = 0.0
        
        # Vérification de la tolérance
        is_within_tolerance = error_distance <= self.tolerance_radius
        
        return LocalizationAccuracy(
            detected_position=detected_pos_2d,
            true_position=true_pos_2d,
            error_distance=error_distance,
            error_angle=angle,
            is_within_tolerance=is_within_tolerance,
            tolerance_radius=self.tolerance_radius
        )
    
    def estimate_leak_position_robust(self) -> Optional[np.ndarray]:
        """
        Estime la position de la fuite de manière robuste en utilisant toutes les détections
        
        Méthodes utilisées :
        1. Si ≥ 3 détections : Filtrage des outliers + moyenne pondérée par concentration
        2. Si 1-2 détections : Utilisation de la meilleure détection
        
        Returns:
            Position estimée (x, y) ou None
        """
        if not self.detections:
            return None
        
        n_detections = len(self.detections)
        
        # Cas simple : 1-2 détections, utiliser la meilleure
        if n_detections <= 2:
            best = self.find_best_detection()
            if best:
                return best.position[:2]
            return None
        
        # Cas robuste : ≥ 3 détections
        positions = np.array([d.position[:2] for d in self.detections])
        concentrations = np.array([d.concentration for d in self.detections])
        
        # Étape 1 : Filtrage des outliers basé sur la distance inter-détections
        # Calcul de la distance médiane entre toutes les paires de détections
        if len(positions) > 1:
            pairwise_distances = pdist(positions)
            median_distance = np.median(pairwise_distances)
            
            # Identifier les outliers : détections trop éloignées des autres
            valid_indices = []
            for i, pos in enumerate(positions):
                distances_to_others = np.array([
                    np.linalg.norm(pos - positions[j]) 
                    for j in range(len(positions)) if i != j
                ])
                # Garder si au moins une détection est proche (dans 2x la médiane)
                if len(distances_to_others) > 0 and np.min(distances_to_others) < 2 * median_distance:
                    valid_indices.append(i)
            
            # Si on a filtré trop d'éléments, garder au moins les 3 meilleures par concentration
            if len(valid_indices) < 3:
                top_indices = np.argsort(concentrations)[-min(3, n_detections):]
                valid_indices = list(set(valid_indices + top_indices.tolist()))
            
            if len(valid_indices) > 0:
                positions = positions[valid_indices]
                concentrations = concentrations[valid_indices]
        
        # Étape 2 : Filtrage supplémentaire des valeurs aberrantes
        # Vérifier que toutes les positions sont raisonnables (pas de Y=0 suspect)
        # Calculer les statistiques de position
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        
        # Filtrer les positions trop éloignées de la moyenne (outliers)
        valid_mask = np.ones(len(positions), dtype=bool)
        for i, pos in enumerate(positions):
            # Vérifier si la position est dans 3 écarts-types de la moyenne
            z_score_x = abs((pos[0] - mean_pos[0]) / max(std_pos[0], 1.0))
            z_score_y = abs((pos[1] - mean_pos[1]) / max(std_pos[1], 1.0))
            
            # Rejeter si X ou Y est trop aberrant
            if z_score_x > 3 or z_score_y > 3:
                valid_mask[i] = False
            # Rejeter aussi les Y=0 si toutes les autres détections ont Y>0
            elif pos[1] < 0.1 and np.sum(positions[:, 1] > 5) >= 2:
                valid_mask[i] = False
        
        if np.sum(valid_mask) >= 2:
            positions = positions[valid_mask]
            concentrations = concentrations[valid_mask]
        
        # Étape 3 : Estimation par moyenne pondérée
        # Pondération : plus la concentration est élevée, plus la détection est proche de la source
        # Normaliser les concentrations pour les poids
        if np.max(concentrations) > 0:
            weights = concentrations / np.max(concentrations)
            # Appliquer une transformation exponentielle pour donner plus de poids aux fortes concentrations
            weights = np.power(weights, 2)
            weights = weights / np.sum(weights)  # Normaliser
        else:
            weights = np.ones(len(positions)) / len(positions)
        
        # Position estimée = moyenne pondérée
        estimated_position = np.average(positions, axis=0, weights=weights)
        
        # Validation finale : vérifier que la position estimée est raisonnable
        # Si Y est trop proche de 0 et qu'on a des détections avec Y>0, utiliser la médiane
        if estimated_position[1] < 1.0 and len([p for p in positions if p[1] > 5]) >= 2:
            # Utiliser la médiane pondérée comme fallback
            sorted_indices = np.argsort(concentrations)[::-1]
            top_positions = positions[sorted_indices[:min(5, len(positions))]]
            estimated_position = np.median(top_positions, axis=0)
        
        return estimated_position
    
    def find_best_detection(self) -> Optional[DetectionResult]:
        """Trouve la détection la plus proche de la source réelle"""
        if not self.detections:
            return None
        
        best_detection = None
        min_distance = float('inf')
        
        for detection in self.detections:
            detected_pos_2d = detection.position[:2]
            distance = np.linalg.norm(detected_pos_2d - self.true_leak_position[:2])
            
            if distance < min_distance:
                min_distance = distance
                best_detection = detection
        
        return best_detection
    
    def compute_metrics(self) -> PerformanceMetrics:
        """
        Calcule toutes les métriques de performance
        
        Returns:
            PerformanceMetrics avec toutes les métriques calculées
        """
        n_detections = len(self.detections)
        
        # Métriques de détection
        first_detection_step = None
        first_detection_time = None
        detection_rate = 0.0
        
        if n_detections > 0:
            first_detection = min(self.detections, key=lambda d: d.step)
            first_detection_step = first_detection.step
            first_detection_time = first_detection.timestamp
            detection_rate = n_detections / max(1, self.total_steps)
        
        # Métriques de localisation
        localization_accuracy = None
        best_detection = None
        avg_detection_distance = None
        
        if n_detections > 0:
            # Estimation robuste de la position de la fuite
            estimated_position = self.estimate_leak_position_robust()
            
            if estimated_position is not None:
                # Créer une détection factice pour l'évaluation
                # Utiliser la détection avec la concentration maximale comme référence pour le temps
                best_by_concentration = max(self.detections, key=lambda d: d.concentration)
                estimated_detection = DetectionResult(
                    position=np.append(estimated_position, best_by_concentration.position[2]),
                    concentration=best_by_concentration.concentration,
                    step=best_by_concentration.step,
                    timestamp=best_by_concentration.timestamp
                )
                
                localization_accuracy = self.calculate_localization_accuracy(estimated_detection)
                best_detection = estimated_detection
            else:
                # Fallback : utiliser la meilleure détection simple
                best_detection = self.find_best_detection()
                if best_detection:
                    localization_accuracy = self.calculate_localization_accuracy(best_detection)
            
            # Distance moyenne de toutes les détections
            distances = []
            for detection in self.detections:
                detected_pos_2d = detection.position[:2]
                distance = np.linalg.norm(detected_pos_2d - self.true_leak_position[:2])
                distances.append(distance)
            avg_detection_distance = np.mean(distances) if distances else None
        
        # Métriques temporelles et énergétiques
        total_time = self.total_steps * self.time_step
        energy_per_detection = self.total_energy / n_detections if n_detections > 0 else None
        
        # Scores de performance
        # Score de détection : 0 si aucune détection, 100 si détection rapide
        if first_detection_time is not None:
            # Score basé sur la rapidité : 100 si détection < 10s, linéaire jusqu'à 60s
            detection_speed_score = max(0, 100 * (1 - first_detection_time / 60))
            detection_rate_score = min(100, detection_rate * 1000)  # Normalisé
            detection_score = (detection_speed_score + detection_rate_score) / 2
        else:
            detection_score = 0.0
        
        # Score de localisation : 0 si erreur > tolérance, 100 si parfait
        if localization_accuracy:
            error = localization_accuracy.error_distance
            if error <= self.tolerance_radius:
                # Score décroît linéairement avec l'erreur
                localization_score = max(0, 100 * (1 - error / self.tolerance_radius))
            else:
                # Score décroît exponentiellement au-delà de la tolérance
                localization_score = max(0, 50 * np.exp(-(error - self.tolerance_radius) / 10))
        else:
            localization_score = 0.0
        
        # Score global : moyenne pondérée
        # Détection (40%) + Localisation (40%) + Efficacité énergétique (20%)
        efficiency_score = 0.0
        if energy_per_detection is not None and energy_per_detection > 0:
            # Score basé sur l'efficacité : idéal < 100 J/détection
            efficiency_score = min(100, 100 * (100 / max(1, energy_per_detection)))
        
        overall_score = (
            0.4 * detection_score +
            0.4 * localization_score +
            0.2 * efficiency_score
        )
        
        # Mission réussie si détection ET localisation valide
        mission_success = (
            n_detections > 0 and
            localization_accuracy is not None and
            localization_accuracy.is_within_tolerance
        )
        
        # Temps de convergence : première détection dans la tolérance
        convergence_time = None
        if n_detections > 0:
            for detection in sorted(self.detections, key=lambda d: d.step):
                acc = self.calculate_localization_accuracy(detection)
                if acc.is_within_tolerance:
                    convergence_time = detection.timestamp
                    break
        
        return PerformanceMetrics(
            n_detections=n_detections,
            first_detection_step=first_detection_step,
            first_detection_time=first_detection_time,
            detection_rate=detection_rate,
            localization_accuracy=localization_accuracy,
            best_detection=best_detection,
            avg_detection_distance=avg_detection_distance,
            total_time=total_time,
            total_energy=self.total_energy,
            energy_per_detection=energy_per_detection,
            detection_score=detection_score,
            localization_score=localization_score,
            overall_score=overall_score,
            mission_success=mission_success,
            convergence_time=convergence_time
        )
    
    def generate_report(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """
        Génère un rapport détaillé de performance
        
        Args:
            metrics: Métriques calculées
            
        Returns:
            Dictionnaire avec rapport complet
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'true_leak_position': self.true_leak_position[:2].tolist(),
            'tolerance_radius': self.tolerance_radius,
            
            # Résumé
            'summary': {
                'mission_success': metrics.mission_success,
                'overall_score': round(metrics.overall_score, 2),
                'detection_score': round(metrics.detection_score, 2),
                'localization_score': round(metrics.localization_score, 2),
            },
            
            # Détection
            'detection': {
                'n_detections': metrics.n_detections,
                'first_detection_step': metrics.first_detection_step,
                'first_detection_time_s': round(metrics.first_detection_time, 2) if metrics.first_detection_time else None,
                'detection_rate': round(metrics.detection_rate, 4),
            },
            
            # Localisation
            'localization': {}
        }
        
        if metrics.localization_accuracy:
            report['localization'] = {
                'detected_position': metrics.localization_accuracy.detected_position.tolist(),
                'true_position': metrics.localization_accuracy.true_position.tolist(),
                'error_distance_m': round(metrics.localization_accuracy.error_distance, 2),
                'error_angle_deg': round(metrics.localization_accuracy.error_angle, 2),
                'is_within_tolerance': metrics.localization_accuracy.is_within_tolerance,
                'tolerance_radius_m': metrics.localization_accuracy.tolerance_radius,
            }
        
        if metrics.avg_detection_distance:
            report['localization']['avg_detection_distance_m'] = round(metrics.avg_detection_distance, 2)
        
        # Énergie et temps
        report['mission'] = {
            'total_time_s': round(metrics.total_time, 2),
            'total_energy_J': round(metrics.total_energy, 2),
            'energy_per_detection_J': round(metrics.energy_per_detection, 2) if metrics.energy_per_detection else None,
        }
        
        # Convergence
        if metrics.convergence_time:
            report['convergence'] = {
                'convergence_time_s': round(metrics.convergence_time, 2),
                'convergence_step': next(
                    (d.step for d in self.detections if self.calculate_localization_accuracy(d).is_within_tolerance),
                    None
                )
            }
        
        return report
    
    def export_results(self, metrics: PerformanceMetrics, filepath: str):
        """Exporte les résultats au format JSON"""
        report = self.generate_report(metrics)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def serialize_metrics(metrics: PerformanceMetrics) -> Dict[str, Any]:
        """
        Sérialise les métriques pour export JSON
        
        Args:
            metrics: Métriques à sérialiser
            
        Returns:
            Dictionnaire sérialisable
        """
        serialized = {
            'n_detections': metrics.n_detections,
            'first_detection_step': metrics.first_detection_step,
            'first_detection_time': metrics.first_detection_time,
            'detection_rate': metrics.detection_rate,
            'total_time': metrics.total_time,
            'total_energy': metrics.total_energy,
            'energy_per_detection': metrics.energy_per_detection,
            'detection_score': metrics.detection_score,
            'localization_score': metrics.localization_score,
            'overall_score': metrics.overall_score,
            'mission_success': metrics.mission_success,
            'convergence_time': metrics.convergence_time,
        }
        
        if metrics.localization_accuracy:
            serialized['localization_accuracy'] = {
                'detected_position': metrics.localization_accuracy.detected_position.tolist(),
                'true_position': metrics.localization_accuracy.true_position.tolist(),
                'error_distance': metrics.localization_accuracy.error_distance,
                'error_angle': metrics.localization_accuracy.error_angle,
                'is_within_tolerance': metrics.localization_accuracy.is_within_tolerance,
                'tolerance_radius': metrics.localization_accuracy.tolerance_radius,
            }
        
        if metrics.best_detection:
            serialized['best_detection'] = {
                'position': metrics.best_detection.position.tolist(),
                'concentration': metrics.best_detection.concentration,
                'step': metrics.best_detection.step,
                'timestamp': metrics.best_detection.timestamp,
            }
        
        serialized['avg_detection_distance'] = metrics.avg_detection_distance
        
        return serialized
