"""
Simulateur 2D simplifié pour HIGHLIGHT+ - Version pour démonstration
Modèle gaussien simple : methane = exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2)) + noise
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


@dataclass
class SimpleConfig:
    """Configuration du simulateur simplifié"""
    # Zone de recherche
    world_size: Tuple[float, float] = (100.0, 100.0)  # (width, height)
    
    # Fuite de méthane
    leak_position: Tuple[float, float] = (50.0, 50.0)
    leak_intensity: float = 1.0  # Amplitude du hotspot
    sigma: float = 8.0  # Largeur du panache gaussien
    
    # Drone
    initial_position: Tuple[float, float] = (10.0, 10.0)
    max_speed: float = 5.0  # m/s
    detection_threshold: float = 0.3  # Seuil de détection (concentration normalisée)
    
    # Simulation
    max_steps: int = 300
    time_step: float = 0.5  # s
    
    # Bruit
    noise_level: float = 0.05  # Bruit gaussien sur les mesures


class SimpleMethanePlume:
    """Modèle de panache simplifié (gaussien 2D)"""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
    
    def concentration(self, x: float, y: float, noise: bool = True) -> float:
        """
        Concentration de méthane à la position (x, y)
        
        Args:
            x, y: Position du drone
            noise: Ajouter du bruit aux mesures
            
        Returns:
            Concentration normalisée [0, 1]
        """
        # Modèle gaussien simple
        dx = x - self.config.leak_position[0]
        dy = y - self.config.leak_position[1]
        
        # Concentration = exp(-distance²/(2*sigma²))
        distance_sq = dx**2 + dy**2
        concentration = self.config.leak_intensity * np.exp(
            -distance_sq / (2 * self.config.sigma**2)
        )
        
        # Normaliser (max = leak_intensity)
        concentration = concentration / self.config.leak_intensity
        
        # Ajouter du bruit de mesure
        if noise:
            concentration += np.random.normal(0, self.config.noise_level)
            concentration = np.clip(concentration, 0, 1)
        
        return concentration
    
    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        Gradient de concentration (direction vers la source)
        
        Returns:
            (grad_x, grad_y) : direction du gradient
        """
        dx = x - self.config.leak_position[0]
        dy = y - self.config.leak_position[1]
        
        # Gradient = -concentration * (dx/sigma², dy/sigma²)
        conc = self.concentration(x, y, noise=False)
        if conc > 1e-6:
            grad_x = -conc * dx / (self.config.sigma**2)
            grad_y = -conc * dy / (self.config.sigma**2)
        else:
            # Direction vers la source même si concentration très faible
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 1e-6:
                grad_x = -dx / dist * 0.01
                grad_y = -dy / dist * 0.01
            else:
                grad_x, grad_y = 0.0, 0.0
        
        return grad_x, grad_y


class SimpleDrone:
    """Drone simplifié avec position et trajectoire"""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.position = np.array(config.initial_position, dtype=np.float32)
        self.trajectory = [self.position.copy()]
        self.detections = []
        self.energy_consumed = 0.0
        self.step_count = 0
    
    def move(self, direction: np.ndarray, speed: Optional[float] = None):
        """
        Déplacer le drone
        
        Args:
            direction: Vecteur direction normalisé
            speed: Vitesse (si None, utilise max_speed)
        """
        if speed is None:
            speed = self.config.max_speed
        
        # Normaliser la direction
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            direction = direction / direction_norm
        
        # Calculer le déplacement
        dt = self.config.time_step
        displacement = direction * speed * dt
        
        # Nouvelle position
        new_position = self.position + displacement
        
        # Contraintes : rester dans la zone
        new_position[0] = np.clip(new_position[0], 0, self.config.world_size[0])
        new_position[1] = np.clip(new_position[1], 0, self.config.world_size[1])
        
        # Consommation énergétique (proportionnelle à la vitesse²)
        energy_step = 0.5 * speed**2 * dt
        self.energy_consumed += energy_step
        
        # Mise à jour
        self.position = new_position
        self.trajectory.append(self.position.copy())
        self.step_count += 1
    
    def detect(self, concentration: float) -> bool:
        """
        Vérifier si une fuite est détectée
        
        Returns:
            True si concentration > seuil
        """
        if concentration > self.config.detection_threshold:
            self.detections.append({
                'position': self.position.copy(),
                'concentration': concentration,
                'step': self.step_count
            })
            return True
        return False
    
    def distance_to_leak(self) -> float:
        """Distance au point de fuite"""
        dx = self.position[0] - self.config.leak_position[0]
        dy = self.position[1] - self.config.leak_position[1]
        return np.sqrt(dx**2 + dy**2)


class NaiveAgent:
    """
    Agent "naïve" : trajectoire en zigzag systématique
    Baseline pour comparaison
    """
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.current_angle = 0.0  # Angle actuel de déplacement
        self.angle_increment = np.pi / 6  # Incrément d'angle (zigzag)
    
    def get_action(self, drone: SimpleDrone) -> np.ndarray:
        """
        Retourne la direction de mouvement (zigzag)
        
        Returns:
            Vecteur direction normalisé
        """
        # Zigzag : changer d'angle périodiquement
        if drone.step_count % 20 == 0:
            self.current_angle += self.angle_increment
        
        # Direction basée sur l'angle
        direction = np.array([
            np.cos(self.current_angle),
            np.sin(self.current_angle)
        ])
        
        return direction


class HighlightAgent:
    """
    Agent HIGHLIGHT+ : utilise le gradient pour naviguer vers la source
    Version simplifiée du Teacher-Student
    """
    
    def __init__(self, config: SimpleConfig, plume: SimpleMethanePlume):
        self.config = config
        self.plume = plume
        self.observations = []  # Historique des observations (pour "Teacher")
        self.exploration_factor = 0.2  # Exploration aléatoire
    
    def get_action(self, drone: SimpleDrone) -> np.ndarray:
        """
        Retourne la direction de mouvement (guidée par gradient)
        
        Returns:
            Vecteur direction normalisé
        """
        # Mesurer la concentration actuelle
        concentration = self.plume.concentration(
            drone.position[0], 
            drone.position[1], 
            noise=True
        )
        
        # Enregistrer l'observation
        self.observations.append({
            'position': drone.position.copy(),
            'concentration': concentration,
            'step': drone.step_count
        })
        
        # Calculer le gradient (direction vers la source)
        grad_x, grad_y = self.plume.gradient(
            drone.position[0], 
            drone.position[1]
        )
        
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        if grad_magnitude > 1e-6:
            # Suivre le gradient (vers la source)
            gradient_direction = np.array([grad_x, grad_y]) / grad_magnitude
            
            # Ajouter de l'exploration
            exploration = np.random.uniform(
                -self.exploration_factor, 
                self.exploration_factor, 
                2
            )
            
            direction = gradient_direction + exploration
            direction = direction / np.linalg.norm(direction)
        else:
            # Pas de gradient : exploration aléatoire
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Réduire l'exploration au fur et à mesure qu'on apprend
        if len(self.observations) > 50:
            self.exploration_factor = max(0.05, self.exploration_factor * 0.99)
        
        return direction


class SimpleSimulator:
    """Simulateur principal"""
    
    def __init__(self, config: SimpleConfig, agent_type: str = "highlight"):
        """
        Args:
            agent_type: "naive" ou "highlight"
        """
        self.config = config
        self.plume = SimpleMethanePlume(config)
        self.drone = SimpleDrone(config)
        
        if agent_type == "naive":
            self.agent = NaiveAgent(config)
        else:
            self.agent = HighlightAgent(config, self.plume)
        
        self.agent_type = agent_type
    
    def run(self) -> dict:
        """
        Exécuter la simulation complète
        
        Returns:
            Dictionnaire avec les métriques de performance
        """
        detection_time = None
        first_detection_step = None
        
        for step in range(self.config.max_steps):
            # Mesurer la concentration
            concentration = self.plume.concentration(
                self.drone.position[0],
                self.drone.position[1],
                noise=True
            )
            
            # Détection
            if self.drone.detect(concentration) and detection_time is None:
                detection_time = step * self.config.time_step
                first_detection_step = step
            
            # Action de l'agent
            direction = self.agent.get_action(self.drone)
            
            # Vitesse adaptative (plus rapide si on est loin)
            distance_to_leak = self.drone.distance_to_leak()
            if distance_to_leak > 50.0:
                speed = self.config.max_speed
            else:
                speed = self.config.max_speed * 0.7
            
            # Déplacer le drone
            self.drone.move(direction, speed)
        
        # Calcul des métriques
        total_time = self.config.max_steps * self.config.time_step
        detection_rate = len(self.drone.detections) / max(1, self.config.max_steps) * 100
        final_distance = self.drone.distance_to_leak()
        
        return {
            'agent_type': self.agent_type,
            'detection_time': detection_time,
            'first_detection_step': first_detection_step,
            'n_detections': len(self.drone.detections),
            'detection_rate': detection_rate,
            'energy_consumed': self.drone.energy_consumed,
            'total_time': total_time,
            'final_distance': final_distance,
            'trajectory': np.array(self.drone.trajectory),
            'detections': self.drone.detections
        }
    
    def visualize_trajectory(self, ax=None, show_leak=True):
        """
        Visualiser la trajectoire
        
        Args:
            ax: Axe matplotlib (si None, crée une nouvelle figure)
            show_leak: Afficher la position de la fuite
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Carte de concentration (fond)
        x = np.linspace(0, self.config.world_size[0], 100)
        y = np.linspace(0, self.config.world_size[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j, i] = self.plume.concentration(X[j, i], Y[j, i], noise=False)
        
        ax.contourf(X, Y, Z, levels=20, cmap='YlOrRd', alpha=0.3)
        ax.set_xlim(0, self.config.world_size[0])
        ax.set_ylim(0, self.config.world_size[1])
        ax.set_aspect('equal')
        
        # Trajectoire
        trajectory = np.array(self.drone.trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
               'b-', linewidth=2, label='Trajectoire', alpha=0.7)
        
        # Point de départ
        ax.plot(trajectory[0, 0], trajectory[0, 1], 
               'gs', markersize=12, label='Départ')
        
        # Point d'arrivée
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 
               'rs', markersize=12, label='Arrivée')
        
        # Détections
        if self.drone.detections:
            detections_pos = np.array([d['position'] for d in self.drone.detections])
            ax.scatter(detections_pos[:, 0], detections_pos[:, 1], 
                      c='red', s=50, marker='*', label='Détections', zorder=5)
        
        # Position de la fuite
        if show_leak:
            ax.plot(self.config.leak_position[0], self.config.leak_position[1], 
                   'rx', markersize=20, linewidth=3, label='Fuite réelle', zorder=6)
        
        ax.set_xlabel('Position X (m)', fontsize=12)
        ax.set_ylabel('Position Y (m)', fontsize=12)
        ax.set_title(f'Trajectoire - Agent {self.agent_type.upper()}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax


if __name__ == "__main__":
    # Test rapide
    config = SimpleConfig(
        leak_position=(60.0, 60.0),
        initial_position=(15.0, 15.0),
        max_steps=200
    )
    
    print(" Test simulateur simplifié HIGHLIGHT+")
    print("=" * 50)
    
    # Test agent naïve
    print("\nTest agent NAIVE...")
    sim_naive = SimpleSimulator(config, agent_type="naive")
    results_naive = sim_naive.run()
    print(f"  Détections: {results_naive['n_detections']}")
    print(f"  Temps de détection: {results_naive['detection_time']:.1f}s")
    print(f"  Énergie: {results_naive['energy_consumed']:.1f} unités")
    
    # Test agent HIGHLIGHT+
    print("\nTest agent HIGHLIGHT+...")
    sim_highlight = SimpleSimulator(config, agent_type="highlight")
    results_highlight = sim_highlight.run()
    print(f"  Détections: {results_highlight['n_detections']}")
    print(f"  Temps de détection: {results_highlight['detection_time']:.1f}s")
    print(f"  Énergie: {results_highlight['energy_consumed']:.1f} unités")
    
    print("\n" + "=" * 50)
    print("Tests termines !")

