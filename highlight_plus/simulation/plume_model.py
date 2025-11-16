"""
Modèle mathématique du panache de méthane pour HIGHLIGHT+
Implémentation de la diffusion atmosphérique avec conditions de vent
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class PlumeConfig:
    """Configuration du modèle de panache"""
    leak_x: float = 50.0
    leak_y: float = 50.0
    leak_intensity: float = 1.0  # kg/s
    wind_speed: float = 2.0      # m/s
    wind_direction: float = 45   # degrés
    sigma_x: float = 5.0         # m
    sigma_y: float = 3.0         # m
    decay_rate: float = 0.01     # s^-1
    temperature: float = 288.15  # K
    pressure: float = 101325     # Pa


class MethanePlume:
    """
    Modèle de panache de méthane basé sur l'équation d'advection-diffusion
    
    Le modèle implémente la solution analytique de l'équation de diffusion
    avec advection par le vent, donnant la concentration C(x,y,t) :
    
    C(x,y,t) = (Q / (2π σ_x σ_y u)) * exp(-((x-x₀)²/(2σ_x²) + (y-y₀)²/(2σ_y²)))
    
    où :
    - Q : débit de la fuite (kg/s)
    - σ_x, σ_y : écarts-types de diffusion
    - u : vitesse du vent
    - (x₀, y₀) : position de la source
    """
    
    def __init__(self, config: PlumeConfig):
        self.config = config
        self._wind_vector = self._compute_wind_vector()
        
    def _compute_wind_vector(self) -> Tuple[float, float]:
        """Calcule le vecteur vent en coordonnées cartésiennes"""
        angle_rad = np.radians(self.config.wind_direction)
        vx = self.config.wind_speed * np.cos(angle_rad)
        vy = self.config.wind_speed * np.sin(angle_rad)
        return vx, vy
    
    def concentration(self, x: np.ndarray, y: np.ndarray, 
                     time: float = 0.0) -> np.ndarray:
        """
        Calcule la concentration de méthane en un point (x,y) à l'instant t
        
        Args:
            x, y: Coordonnées spatiales (m)
            time: Temps depuis le début de la fuite (s)
            
        Returns:
            Concentration de méthane (kg/m³)
        """
        # Conversion en arrays numpy si nécessaire
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Position effective de la source (advection par le vent)
        vx, vy = self._wind_vector
        effective_x = self.config.leak_x + vx * time
        effective_y = self.config.leak_y + vy * time
        
        # Calcul des distances relatives
        dx = x - effective_x
        dy = y - effective_y
        
        # Facteur de décroissance temporelle
        decay_factor = np.exp(-self.config.decay_rate * time)
        
        # Calcul de la concentration selon le modèle gaussien
        # C = (Q / (2π σ_x σ_y u)) * exp(-(dx²/(2σ_x²) + dy²/(2σ_y²)))
        wind_speed = np.sqrt(vx**2 + vy**2)
        if wind_speed < 0.1:  # Éviter la division par zéro
            wind_speed = 0.1
            
        normalization = (self.config.leak_intensity * decay_factor) / \
                       (2 * np.pi * self.config.sigma_x * self.config.sigma_y * wind_speed)
        
        exponent = -(dx**2 / (2 * self.config.sigma_x**2) + 
                    dy**2 / (2 * self.config.sigma_y**2))
        
        concentration = normalization * np.exp(exponent)
        
        return concentration
    
    def gradient(self, x: np.ndarray, y: np.ndarray, 
                time: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule le gradient de concentration ∇C = (∂C/∂x, ∂C/∂y)
        
        Args:
            x, y: Coordonnées spatiales (m)
            time: Temps depuis le début de la fuite (s)
            
        Returns:
            Tuple (grad_x, grad_y) : Composantes du gradient
        """
        # Position effective de la source
        vx, vy = self._wind_vector
        effective_x = self.config.leak_x + vx * time
        effective_y = self.config.leak_y + vy * time
        
        # Calcul des distances relatives (AVANT le calcul de C pour éviter division par zéro)
        dx = x - effective_x
        dy = y - effective_y
        
        # Calcul de la concentration
        C = self.concentration(x, y, time)
        
        # Dérivées partielles : grad = -C * (dx/sigma², dy/sigma²)
        grad_x = -C * dx / (self.config.sigma_x**2)
        grad_y = -C * dy / (self.config.sigma_y**2)
        
        # Amélioration : Si la concentration est très faible (loin de la source),
        # utiliser un gradient directionnel basé sur la distance pour guider vers la source
        # Cela évite que le gradient soit ≈0 et bloque la navigation
        concentration_threshold = 1e-6
        
        # Traitement selon le type (scalar ou array)
        if isinstance(C, np.ndarray):
            low_concentration_mask = C < concentration_threshold
            if np.any(low_concentration_mask):
                # Pour les points avec faible concentration, utiliser un gradient directionnel
                distance_to_source = np.sqrt(dx**2 + dy**2)
                # Éviter division par zéro
                safe_distance = np.where(distance_to_source > 1e-6, distance_to_source, 1.0)
                # Normaliser pour donner une direction vers la source (faible mais non-nul)
                grad_x = np.where(low_concentration_mask, 
                                 -dx / safe_distance * 0.01, 
                                 grad_x)
                grad_y = np.where(low_concentration_mask,
                                 -dy / safe_distance * 0.01,
                                 grad_y)
        else:
            # Scalar case
            if C < concentration_threshold:
                distance_to_source = np.sqrt(dx**2 + dy**2)
                if distance_to_source > 1e-6:
                    # Gradient directionnel vers la source (faible mais non-nul)
                    # Permet la navigation même quand C≈0
                    grad_x = -dx / distance_to_source * 0.01
                    grad_y = -dy / distance_to_source * 0.01
        
        return grad_x, grad_y
    
    def create_concentration_map(self, x_range: Tuple[float, float], 
                               y_range: Tuple[float, float],
                               resolution: int = 100,
                               time: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Crée une carte de concentration sur une grille régulière
        
        Args:
            x_range: (x_min, x_max) en mètres
            y_range: (y_min, y_max) en mètres
            resolution: Nombre de points par dimension
            time: Temps depuis le début de la fuite (s)
            
        Returns:
            Tuple (X, Y, C) : Grilles de coordonnées et concentration
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        C = self.concentration(X, Y, time)
        
        return X, Y, C
    
    def plot_plume(self, x_range: Tuple[float, float] = (0, 100),
                  y_range: Tuple[float, float] = (0, 100),
                  resolution: int = 100,
                  time: float = 0.0,
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Visualise le panache de méthane
        
        Args:
            x_range, y_range: Limites de la zone à visualiser
            resolution: Résolution de la grille
            time: Temps depuis le début de la fuite
            ax: Axes matplotlib existants (optionnel)
            
        Returns:
            Axes matplotlib avec la visualisation
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Création de la carte de concentration
        X, Y, C = self.create_concentration_map(x_range, y_range, resolution, time)
        
        # Visualisation
        im = ax.contourf(X, Y, C, levels=20, cmap='viridis', alpha=0.8)
        ax.contour(X, Y, C, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Marquer la source
        ax.plot(self.config.leak_x, self.config.leak_y, 'r*', 
               markersize=15, label='Source de fuite')
        
        # Vecteur vent
        vx, vy = self._wind_vector
        ax.arrow(self.config.leak_x, self.config.leak_y, 
                vx * 5, vy * 5, head_width=2, head_length=1, 
                fc='red', ec='red', alpha=0.7, label='Vent')
        
        # Configuration de l'affichage
        ax.set_xlabel('Position X (m)')
        ax.set_ylabel('Position Y (m)')
        ax.set_title(f'Panache de méthane - t = {time:.1f}s')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Barre de couleur
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Concentration (kg/m³)')
        
        return ax


def create_test_plume() -> MethanePlume:
    """Crée un panache de test avec des paramètres réalistes"""
    config = PlumeConfig(
        leak_x=50.0,
        leak_y=50.0,
        leak_intensity=0.5,  # kg/s
        wind_speed=3.0,      # m/s
        wind_direction=30,   # degrés
        sigma_x=8.0,         # m
        sigma_y=5.0,         # m
        decay_rate=0.005     # s^-1
    )
    return MethanePlume(config)


if __name__ == "__main__":
    # Test du modèle
    plume = create_test_plume()
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Différents instants
    times = [0, 10, 20, 30]
    for i, t in enumerate(times):
        ax = axes[i//2, i%2]
        plume.plot_plume(time=t, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    # Test du gradient
    x_test = np.array([45, 50, 55])
    y_test = np.array([50, 50, 50])
    grad_x, grad_y = plume.gradient(x_test, y_test)
    
    print("Test du gradient:")
    for i in range(len(x_test)):
        print(f"Point ({x_test[i]}, {y_test[i]}): grad = ({grad_x[i]:.3f}, {grad_y[i]:.3f})")




