"""
Validateur de fuite de méthane basé sur Processus Gaussiens
Intègre la logique de validation GP pour détection robuste de position de fuite
"""

import numpy as np
from typing import Tuple, Optional, List
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import warnings
warnings.filterwarnings('ignore')


class MethaneLeakValidator:
    """
    Validateur de fuite de méthane utilisant un Processus Gaussien
    
    Cette classe accumule les mesures de concentration au fil du temps
    et utilise un GP pour modéliser la carte de concentration, permettant
    d'estimer la position de la fuite avec une probabilité.
    
    Caractéristiques :
    - Modélisation GP de la carte de concentration
    - Estimation probabiliste de la position de fuite
    - Seuil de probabilité configurable
    - Mise à jour incrémentale avec nouvelles mesures
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (100, 100),
                 world_bounds: Tuple[float, float, float, float] = (0, 100, 0, 100),
                 noise: float = 1e-2,
                 threshold_prob: float = 0.95,
                 kernel_length_scale: float = 5.0,
                 kernel_variance: float = 1.0):
        """
        Args:
            grid_size: Taille de la grille pour la recherche (nx, ny)
            world_bounds: Limites du monde (x_min, x_max, y_min, y_max)
            noise: Niveau de bruit du GP
            threshold_prob: Seuil de probabilité pour confirmer une fuite (0-1)
            kernel_length_scale: Échelle de longueur du kernel RBF
            kernel_variance: Variance du kernel
        """
        self.grid_size = grid_size
        self.world_bounds = world_bounds
        self.threshold_prob = threshold_prob
        
        # Stockage des mesures
        self.X: List[np.ndarray] = []  # Positions
        self.Y: List[float] = []  # Concentrations
        
        # Kernel GP
        self.kernel = (C(kernel_variance) * RBF(length_scale=kernel_length_scale) + 
                      WhiteKernel(noise_level=noise**2))
        
        # GP initialisé vide
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=noise**2,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        # Historique des estimations
        self.estimated_positions: List[Tuple[np.ndarray, float]] = []
        
    def add_measurement(self, position: Tuple[float, float], concentration: float):
        """
        Ajoute une nouvelle mesure de concentration
        
        Args:
            position: Position (x, y) de la mesure
            concentration: Concentration mesurée
        """
        self.X.append(np.array(position))
        self.Y.append(float(concentration))
        self._update_gp()
    
    def _update_gp(self):
        """Met à jour le GP avec toutes les mesures accumulées"""
        if len(self.X) >= 2:  # Minimum 2 points pour un GP
            try:
                X_array = np.array(self.X)
                Y_array = np.array(self.Y)
                self.gp.fit(X_array, Y_array)
            except Exception as e:
                # Si erreur, on continue avec les mesures précédentes
                pass
    
    def get_leak_position(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Estime la position de la fuite à partir du GP (VERSION AMÉLIORÉE)
        
        Stratégie améliorée pour détection excellente :
        1. Prédit la concentration ET l'incertitude sur une grille fine
        2. Utilise une combinaison concentration + confiance (faible incertitude)
        3. Filtre les zones avec trop d'incertitude
        4. Identifie les candidats avec score combiné élevé
        5. Retourne le meilleur candidat avec validation multi-critères
        
        Returns:
            (position_estimée, probabilité) ou (None, None) si pas de fuite confirmée
        """
        if len(self.X) < 2:  # Minimum 2 mesures (réduit pour détection plus précoce)
            return None, None
        
        try:
            # Créer la grille de recherche (résolution augmentée pour précision)
            x_min, x_max, y_min, y_max = self.world_bounds
            nx, ny = self.grid_size
            
            # Utiliser une grille plus fine si peu de mesures (meilleure précision)
            if len(self.X) < 10:
                nx, ny = max(nx, 150), max(ny, 150)
            
            xx = np.linspace(x_min, x_max, nx)
            yy = np.linspace(y_min, y_max, ny)
            XX, YY = np.meshgrid(xx, yy)
            grid_points = np.c_[XX.ravel(), YY.ravel()]
            
            # Prédiction GP avec incertitude
            mu, sigma = self.gp.predict(grid_points, return_std=True)
            
            # AMÉLIORATION : Score combiné concentration + confiance (faible incertitude)
            # Normaliser la concentration (0-1)
            mu_min = mu.min()
            mu_max = mu.max()
            if mu_max - mu_min < 1e-6:
                return None, None
            
            mu_normalized = (mu - mu_min) / (mu_max - mu_min + 1e-6)
            
            # Normaliser l'incertitude (0-1, inversé : faible incertitude = haute confiance)
            sigma_max = sigma.max()
            if sigma_max < 1e-6:
                confidence = np.ones_like(sigma)
            else:
                confidence = 1.0 - (sigma / (sigma_max + 1e-6))
                confidence = np.clip(confidence, 0.0, 1.0)
            
            # Score combiné : 70% concentration + 30% confiance (faible incertitude)
            combined_score = 0.7 * mu_normalized + 0.3 * confidence
            
            # AMÉLIORATION : Filtrer les zones avec trop d'incertitude relative
            # Si l'incertitude est > 50% de la concentration, réduire le score
            relative_uncertainty = sigma / (np.abs(mu) + 1e-6)
            uncertainty_penalty = np.where(relative_uncertainty > 0.5, 0.5, 1.0)
            combined_score = combined_score * uncertainty_penalty
            
            # Identifier les candidats au-dessus du seuil (seuil adaptatif)
            # Seuil plus bas si peu de mesures (détection plus précoce)
            adaptive_threshold = self.threshold_prob
            if len(self.X) < 10:
                adaptive_threshold = max(0.6, self.threshold_prob - 0.15)
            
            candidates = np.where(combined_score >= adaptive_threshold)[0]
            
            if len(candidates) == 0:
                # Si aucun candidat au seuil, prendre le maximum du score combiné
                idx_max = np.argmax(combined_score)
                leak_pos = grid_points[idx_max]
                leak_prob = float(combined_score[idx_max])
                
                # Seuil minimum pour confirmer (plus permissif si peu de mesures)
                min_prob = 0.4 if len(self.X) < 10 else 0.5
                if leak_prob < min_prob:
                    return None, None
                
                return leak_pos, leak_prob
            
            # Parmi les candidats, prendre celui avec le score combiné maximal
            candidate_scores = combined_score[candidates]
            idx_max_candidate = candidates[np.argmax(candidate_scores)]
            
            leak_pos = grid_points[idx_max_candidate]
            leak_prob = float(combined_score[idx_max_candidate])
            
            # Stocker l'estimation
            self.estimated_positions.append((leak_pos.copy(), float(leak_prob)))
            
            return leak_pos, float(leak_prob)
            
        except Exception as e:
            # En cas d'erreur, retourner None
            return None, None
    
    def get_all_leak_positions(self, min_probability: float = 0.6, min_distance: float = 5.0) -> List[Tuple[np.ndarray, float]]:
        """
        Extrait TOUTES les positions de fuite avec probabilité élevée de la carte de confiance GP
        
        Args:
            min_probability: Probabilité minimale pour considérer une position (0-1)
            min_distance: Distance minimale entre deux positions détectées (m)
            
        Returns:
            Liste de tuples (position, probabilité) triés par probabilité décroissante
        """
        if len(self.X) < 2:
            return []
        
        try:
            # Créer la grille de recherche
            x_min, x_max, y_min, y_max = self.world_bounds
            nx, ny = self.grid_size
            
            # Utiliser une grille fine pour meilleure précision
            if len(self.X) < 10:
                nx, ny = max(nx, 150), max(ny, 150)
            
            xx = np.linspace(x_min, x_max, nx)
            yy = np.linspace(y_min, y_max, ny)
            XX, YY = np.meshgrid(xx, yy)
            grid_points = np.c_[XX.ravel(), YY.ravel()]
            
            # Prédiction GP avec incertitude
            mu, sigma = self.gp.predict(grid_points, return_std=True)
            
            # Calcul du score combiné (comme dans get_leak_position)
            mu_min = mu.min()
            mu_max = mu.max()
            if mu_max - mu_min < 1e-6:
                return []
            
            mu_normalized = (mu - mu_min) / (mu_max - mu_min + 1e-6)
            
            # Normaliser l'incertitude (0-1, inversé : faible incertitude = haute confiance)
            sigma_max = sigma.max()
            if sigma_max < 1e-6:
                confidence = np.ones_like(sigma)
            else:
                confidence = 1.0 - (sigma / (sigma_max + 1e-6))
                confidence = np.clip(confidence, 0.0, 1.0)
            
            # Score combiné : 70% concentration + 30% confiance
            combined_score = 0.7 * mu_normalized + 0.3 * confidence
            
            # Filtrer les zones avec trop d'incertitude relative
            relative_uncertainty = sigma / (np.abs(mu) + 1e-6)
            uncertainty_penalty = np.where(relative_uncertainty > 0.5, 0.5, 1.0)
            combined_score = combined_score * uncertainty_penalty
            
            # Identifier TOUS les candidats au-dessus du seuil
            candidates = np.where(combined_score >= min_probability)[0]
            
            if len(candidates) == 0:
                # Si aucun candidat au seuil, prendre les meilleurs (top 5)
                top_k = min(5, len(combined_score))
                top_indices = np.argpartition(combined_score, -top_k)[-top_k:]
                candidates = top_indices[combined_score[top_indices] >= 0.4]  # Seuil minimum absolu
            
            if len(candidates) == 0:
                return []
            
            # Extraire les positions et probabilités
            candidate_positions = grid_points[candidates]
            candidate_probs = combined_score[candidates]
            
            # Trier par probabilité décroissante
            sorted_indices = np.argsort(candidate_probs)[::-1]
            candidate_positions = candidate_positions[sorted_indices]
            candidate_probs = candidate_probs[sorted_indices]
            
            # Clustering : regrouper les positions proches et garder la meilleure de chaque groupe
            final_positions = []
            used_indices = set()
            
            for i in range(len(candidate_positions)):
                if i in used_indices:
                    continue
                
                pos_i = candidate_positions[i]
                prob_i = candidate_probs[i]
                
                # Trouver toutes les positions proches
                group = [i]
                for j in range(i + 1, len(candidate_positions)):
                    if j in used_indices:
                        continue
                    pos_j = candidate_positions[j]
                    dist = np.linalg.norm(pos_i - pos_j)
                    if dist < min_distance:
                        group.append(j)
                        used_indices.add(j)
                
                # Prendre la position avec la plus haute probabilité dans le groupe
                group_probs = candidate_probs[group]
                best_in_group_idx = group[np.argmax(group_probs)]
                best_pos = candidate_positions[best_in_group_idx]
                best_prob = candidate_probs[best_in_group_idx]
                
                final_positions.append((best_pos.copy(), float(best_prob)))
                used_indices.update(group)
            
            # Trier par probabilité décroissante
            final_positions.sort(key=lambda x: x[1], reverse=True)
            
            return final_positions
            
        except Exception as e:
            # En cas d'erreur, retourner liste vide
            return []
    
    def get_confidence_map(self, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Génère une carte de confiance (probabilité) sur la grille
        
        Args:
            resolution: Résolution de la grille (resolution x resolution)
            
        Returns:
            (XX, YY, prob_map) - Grille et carte de probabilités
        """
        if len(self.X) < 2:
            x_min, x_max, y_min, y_max = self.world_bounds
            xx = np.linspace(x_min, x_max, resolution)
            yy = np.linspace(y_min, y_max, resolution)
            XX, YY = np.meshgrid(xx, yy)
            return XX, YY, np.zeros_like(XX)
        
        try:
            x_min, x_max, y_min, y_max = self.world_bounds
            xx = np.linspace(x_min, x_max, resolution)
            yy = np.linspace(y_min, y_max, resolution)
            XX, YY = np.meshgrid(xx, yy)
            grid_points = np.c_[XX.ravel(), YY.ravel()]
            
            mu, _ = self.gp.predict(grid_points, return_std=True)
            
            # Normalisation
            mu_min = mu.min()
            mu_max = mu.max()
            if mu_max - mu_min < 1e-6:
                prob_map = np.zeros_like(XX)
            else:
                prob = (mu - mu_min) / (mu_max - mu_min + 1e-6)
                prob_map = prob.reshape(XX.shape)
            
            return XX, YY, prob_map
            
        except Exception:
            x_min, x_max, y_min, y_max = self.world_bounds
            xx = np.linspace(x_min, x_max, resolution)
            yy = np.linspace(y_min, y_max, resolution)
            XX, YY = np.meshgrid(xx, yy)
            return XX, YY, np.zeros_like(XX)
    
    def get_statistics(self) -> dict:
        """Retourne des statistiques sur le validateur"""
        return {
            'n_measurements': len(self.X),
            'n_estimations': len(self.estimated_positions),
            'last_estimation': self.estimated_positions[-1] if self.estimated_positions else None,
            'gp_trained': len(self.X) >= 2
        }
    
    def reset(self):
        """Réinitialise le validateur"""
        self.X = []
        self.Y = []
        self.estimated_positions = []
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.gp.alpha,
            normalize_y=True,
            n_restarts_optimizer=5
        )

