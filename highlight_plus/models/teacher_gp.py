"""
Expert (Teacher) basé sur les Processus Gaussiens pour HIGHLIGHT+
Implémentation de l'algorithme d'apprentissage actif pour la détection de fuites
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


@dataclass
class TeacherConfig:
    """Configuration de l'Expert (Teacher)"""
    # Kernel GP
    kernel_length_scale: float = 10.0
    kernel_variance: float = 1.0
    noise_level: float = 1e-3
    
    # Stratégie d'exploration
    acquisition_function: str = "UCB"  # UCB, EI, PI
    exploration_parameter: float = 2.0  # β pour UCB
    
    # Contraintes de mouvement
    max_step_size: float = 5.0
    min_step_size: float = 1.0
    
    # Critères d'arrêt
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    min_uncertainty: float = 0.01


class GaussianProcessTeacher:
    """
    Expert (Teacher) utilisant les Processus Gaussiens pour l'apprentissage actif
    
    L'Expert utilise un modèle GP pour estimer la carte de concentration de méthane
    et choisit les prochains points à explorer en maximisant une fonction d'acquisition.
    
    Stratégie d'apprentissage actif :
    1. Initialiser le GP avec quelques points aléatoires
    2. Pour chaque itération :
       a. Entraîner le GP sur les observations actuelles
       b. Calculer la fonction d'acquisition sur l'espace de recherche
       c. Choisir le point maximisant l'acquisition
       d. Se déplacer vers ce point et prendre une mesure
       e. Ajouter la nouvelle observation au GP
    3. Répéter jusqu'à convergence ou limite d'itérations
    """
    
    def __init__(self, config: TeacherConfig, world_bounds: Tuple[float, float, float, float]):
        self.config = config
        self.world_bounds = world_bounds  # (x_min, x_max, y_min, y_max)
        
        # Initialisation du GP
        self._initialize_gp()
        
        # Historique des observations
        self.observations = []  # List of (x, y, concentration, uncertainty)
        self.trajectory = []    # List of (x, y) positions
        self.measurements = []  # List of measured concentrations
        
        # État actuel
        self.current_position = None
        self.current_uncertainty = None
        
    def _initialize_gp(self):
        """Initialise le modèle de Processus Gaussien"""
        # Kernel composite : RBF + bruit blanc
        kernel = (ConstantKernel(constant_value=self.config.kernel_variance) * 
                 RBF(length_scale=self.config.kernel_length_scale) + 
                 WhiteKernel(noise_level=self.config.noise_level))
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=10,
            random_state=42
        )
        
    def add_observation(self, x: float, y: float, concentration: float, 
                       uncertainty: Optional[float] = None):
        """
        Ajoute une nouvelle observation au modèle GP
        
        Args:
            x, y: Position de la mesure
            concentration: Concentration mesurée
            uncertainty: Incertitude de la mesure (optionnel)
        """
        self.observations.append((x, y, concentration, uncertainty))
        self.trajectory.append((x, y))
        self.measurements.append(concentration)
        
        # Mise à jour du GP
        self._update_gp()
        
    def _update_gp(self):
        """Met à jour le modèle GP avec toutes les observations"""
        if len(self.observations) < 2:
            return
            
        # Extraction des données
        X = np.array([[obs[0], obs[1]] for obs in self.observations])
        y = np.array([obs[2] for obs in self.observations])
        
        # Entraînement du GP
        self.gp.fit(X, y)
        
    def predict(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédit la concentration et l'incertitude en des points donnés
        
        Args:
            x, y: Coordonnées des points à prédire
            
        Returns:
            Tuple (mean, std) : Moyenne et écart-type des prédictions
        """
        if len(self.observations) < 2:
            # Retourner des valeurs par défaut si pas assez d'observations
            return np.zeros_like(x), np.ones_like(x)
            
        X_pred = np.column_stack([x.ravel(), y.ravel()])
        mean, std = self.gp.predict(X_pred, return_std=True)
        
        return mean.reshape(x.shape), std.reshape(x.shape)
    
    def acquisition_function(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcule la fonction d'acquisition pour guider l'exploration
        
        Args:
            x, y: Grille de points à évaluer
            
        Returns:
            Valeurs de la fonction d'acquisition
        """
        mean, std = self.predict(x, y)
        
        if self.config.acquisition_function == "UCB":
            # Upper Confidence Bound
            return mean + self.config.exploration_parameter * std
        elif self.config.acquisition_function == "EI":
            # Expected Improvement (simplifié)
            return std * np.exp(-0.5 * (mean / std)**2)
        elif self.config.acquisition_function == "PI":
            # Probability of Improvement
            return std
        else:
            # Par défaut : maximiser l'incertitude
            return std
    
    def select_next_point(self, current_x: float, current_y: float, 
                         gradient_x: Optional[float] = None,
                         gradient_y: Optional[float] = None,
                         target_position: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        Sélectionne le prochain point à explorer
        
        Args:
            current_x, current_y: Position actuelle
            gradient_x, gradient_y: Gradient de concentration (optionnel, pour guider l'exploration)
            target_position: Position cible estimée (optionnel, pour exploration initiale)
            
        Returns:
            Tuple (next_x, next_y) : Prochaine position
        """
        # Si on a un gradient significatif, l'utiliser pour guider l'exploration
        if gradient_x is not None and gradient_y is not None:
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            if gradient_magnitude > 1e-6:
                # Normaliser le gradient
                grad_norm_x = gradient_x / gradient_magnitude
                grad_norm_y = gradient_y / gradient_magnitude
                
                # Aller dans la direction du gradient (vers la source)
                # Taille de pas adaptative : plus grand si gradient est fort
                base_step = (self.config.min_step_size + self.config.max_step_size) / 2
                step_size = base_step * (1 + min(gradient_magnitude * 10, 1.0))  # Max 2x si gradient fort
                step_size = np.clip(step_size, self.config.min_step_size, self.config.max_step_size)
                next_x = current_x + step_size * grad_norm_x
                next_y = current_y + step_size * grad_norm_y
                
                # S'assurer que le point est dans les limites
                x_min, x_max, y_min, y_max = self.world_bounds
                next_x = np.clip(next_x, x_min, x_max)
                next_y = np.clip(next_y, y_min, y_max)
                
                return next_x, next_y
        
        # Si pas de gradient mais on a peu d'observations, explorer vers le centre ou la cible
        # Augmenter à 10 observations pour meilleure initialisation
        if len(self.observations) < 10 and target_position is not None:
            # Navigation vers la cible estimée avec exploration
            target_x, target_y = target_position
            direction = np.array([target_x - current_x, target_y - current_y])
            dist_to_target = np.linalg.norm(direction)
            
            if dist_to_target > 1.0:
                # Normaliser et ajouter un peu d'exploration
                direction = direction / dist_to_target
                exploration = np.random.uniform(-0.3, 0.3, 2)
                direction += exploration
                direction = direction / np.linalg.norm(direction)
                
                step_size = min(self.config.max_step_size, dist_to_target * 0.5)
                next_x = current_x + step_size * direction[0]
                next_y = current_y + step_size * direction[1]
                
                x_min, x_max, y_min, y_max = self.world_bounds
                next_x = np.clip(next_x, x_min, x_max)
                next_y = np.clip(next_y, y_min, y_max)
                
                return next_x, next_y
        
        # Sinon, utiliser la méthode classique avec acquisition function
        # Grille de recherche
        x_min, x_max, y_min, y_max = self.world_bounds
        x_grid = np.linspace(x_min, x_max, 50)
        y_grid = np.linspace(y_min, y_max, 50)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Calcul de la fonction d'acquisition
        acquisition_values = self.acquisition_function(X_grid, Y_grid)
        
        # Contrainte de distance maximale
        distances = np.sqrt((X_grid - current_x)**2 + (Y_grid - current_y)**2)
        valid_mask = (distances <= self.config.max_step_size) & \
                    (distances >= self.config.min_step_size)
        
        if not np.any(valid_mask):
            # Si aucun point valide, choisir un point aléatoire dans la contrainte
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(self.config.min_step_size, 
                                       self.config.max_step_size)
            next_x = current_x + distance * np.cos(angle)
            next_y = current_y + distance * np.sin(angle)
            
            # S'assurer que le point est dans les limites
            next_x = np.clip(next_x, x_min, x_max)
            next_y = np.clip(next_y, y_min, y_max)
            
            return next_x, next_y
        
        # Sélectionner le point avec la plus grande valeur d'acquisition
        valid_acquisition = acquisition_values.copy()
        valid_acquisition[~valid_mask] = -np.inf
        
        max_idx = np.unravel_index(np.argmax(valid_acquisition), valid_acquisition.shape)
        next_x = X_grid[max_idx]
        next_y = Y_grid[max_idx]
        
        return next_x, next_y
    
    def get_uncertainty_map(self, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Génère une carte d'incertitude du modèle GP
        
        Args:
            resolution: Résolution de la grille
            
        Returns:
            Tuple (X, Y, uncertainty) : Grille et carte d'incertitude
        """
        x_min, x_max, y_min, y_max = self.world_bounds
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        _, uncertainty = self.predict(X, Y)
        
        return X, Y, uncertainty
    
    def get_concentration_map(self, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Génère une carte de concentration prédite
        
        Args:
            resolution: Résolution de la grille
            
        Returns:
            Tuple (X, Y, concentration) : Grille et carte de concentration
        """
        x_min, x_max, y_min, y_max = self.world_bounds
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        concentration, _ = self.predict(X, Y)
        
        return X, Y, concentration
    
    def plot_results(self, true_plume=None, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Visualise les résultats de l'Expert
        
        Args:
            true_plume: Modèle de panache réel (optionnel)
            ax: Axes matplotlib existants (optionnel)
            
        Returns:
            Axes matplotlib avec la visualisation
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        # Carte de concentration prédite
        X, Y, concentration = self.get_concentration_map()
        
        # Visualisation de la prédiction
        im = ax.contourf(X, Y, concentration, levels=20, cmap='viridis', alpha=0.7)
        ax.contour(X, Y, concentration, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Trajectoire de l'Expert
        if len(self.trajectory) > 1:
            traj_x = [pos[0] for pos in self.trajectory]
            traj_y = [pos[1] for pos in self.trajectory]
            ax.plot(traj_x, traj_y, 'r-', linewidth=2, label='Trajectoire Expert')
            ax.scatter(traj_x, traj_y, c='red', s=50, zorder=5)
            
            # Marquer le point de départ et d'arrivée
            ax.scatter(traj_x[0], traj_y[0], c='green', s=100, marker='o', 
                      label='Départ', zorder=6)
            ax.scatter(traj_x[-1], traj_y[-1], c='blue', s=100, marker='s', 
                      label='Arrivée', zorder=6)
        
        # Panache réel (si fourni)
        if true_plume is not None:
            true_X, true_Y, true_C = true_plume.create_concentration_map(
                (self.world_bounds[0], self.world_bounds[1]),
                (self.world_bounds[2], self.world_bounds[3])
            )
            ax.contour(true_X, true_Y, true_C, levels=5, colors='white', 
                      linestyles='--', alpha=0.8, linewidths=2)
        
        # Configuration
        ax.set_xlabel('Position X (m)')
        ax.set_ylabel('Position Y (m)')
        ax.set_title('Expert (Teacher) - Prédiction et Trajectoire')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Barre de couleur
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Concentration prédite (kg/m³)')
        
        return ax
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calcule les métriques de performance de l'Expert
        
        Returns:
            Dictionnaire avec les métriques
        """
        if len(self.observations) < 2:
            return {}
        
        # Nombre total d'observations
        n_observations = len(self.observations)
        
        # Distance totale parcourue
        total_distance = 0.0
        for i in range(1, len(self.trajectory)):
            dx = self.trajectory[i][0] - self.trajectory[i-1][0]
            dy = self.trajectory[i][1] - self.trajectory[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        # Incertitude moyenne
        mean_uncertainty = np.mean([obs[3] for obs in self.observations if obs[3] is not None])
        
        # Concentration maximale détectée
        max_concentration = max(self.measurements)
        
        return {
            'n_observations': n_observations,
            'total_distance': total_distance,
            'mean_uncertainty': mean_uncertainty if mean_uncertainty is not None else 0.0,
            'max_concentration': max_concentration,
            'efficiency': max_concentration / total_distance if total_distance > 0 else 0.0
        }
    
    def reset(self):
        """Remet à zéro l'Expert"""
        self.observations = []
        self.trajectory = []
        self.measurements = []
        self.current_position = None
        self.current_uncertainty = None
        self._initialize_gp()


def create_test_teacher() -> GaussianProcessTeacher:
    """Crée un Expert de test avec des paramètres réalistes"""
    config = TeacherConfig(
        kernel_length_scale=8.0,
        kernel_variance=1.0,
        exploration_parameter=2.5,
        max_step_size=8.0,
        min_step_size=2.0
    )
    
    world_bounds = (0, 100, 0, 100)  # 100x100 m
    return GaussianProcessTeacher(config, world_bounds)


if __name__ == "__main__":
    # Test de l'Expert
    teacher = create_test_teacher()
    
    # Simulation d'observations
    np.random.seed(42)
    
    # Position initiale
    current_x, current_y = 10.0, 10.0
    teacher.current_position = (current_x, current_y)
    
    print("Test de l'Expert (Teacher):")
    print("=" * 40)
    
    # Simulation de quelques étapes
    for i in range(10):
        # Mesure simulée (concentration aléatoire)
        concentration = np.random.exponential(0.1)
        uncertainty = np.random.uniform(0.05, 0.2)
        
        # Ajout de l'observation
        teacher.add_observation(current_x, current_y, concentration, uncertainty)
        
        # Sélection du prochain point
        next_x, next_y = teacher.select_next_point(current_x, current_y)
        
        print(f"Étape {i+1}: Position ({current_x:.1f}, {current_y:.1f}) -> "
              f"Conc: {concentration:.3f}, Next: ({next_x:.1f}, {next_y:.1f})")
        
        current_x, current_y = next_x, next_y
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    teacher.plot_results()
    plt.show()
    
    # Métriques
    metrics = teacher.get_performance_metrics()
    print("\nMétriques de performance:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")




