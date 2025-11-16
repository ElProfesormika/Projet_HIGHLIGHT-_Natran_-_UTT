"""
Environnement de simulation Gymnasium pour HIGHLIGHT+
Intègre le panache de méthane, le capteur TDLAS et les contraintes du drone
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Import des composants du projet
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.plume_model import MethanePlume, PlumeConfig
from sensors.tdlas_sensor import TDLASSensor, TDLASConfig


@dataclass
class EnvironmentConfig:
    """Configuration de l'environnement de simulation"""
    # Dimensions du monde
    world_size: Tuple[float, float] = (100.0, 100.0)  # (width, height) en mètres
    
    # Paramètres temporels
    time_step: float = 0.1  # Pas de temps (s)
    max_steps: int = 1000   # Nombre maximum d'étapes
    
    # Conditions initiales
    initial_position: Tuple[float, float] = (10.0, 10.0)
    initial_altitude: float = 5.0
    
    # Contraintes du drone
    max_speed: float = 5.0      # m/s
    max_altitude: float = 20.0  # m
    min_altitude: float = 2.0   # m
    
    # Consommation énergétique
    base_power: float = 100.0        # W
    speed_coefficient: float = 50.0  # W/(m/s)
    altitude_coefficient: float = 25.0  # W/m
    
    # Fonction de récompense
    detection_bonus: float = 10.0
    energy_penalty: float = -0.1
    boundary_penalty: float = -5.0
    exploration_bonus: float = 1.0


class MethaneDetectionEnv(gym.Env):
    """
    Environnement de simulation pour la détection de fuites de méthane
    
    L'environnement simule un drone équipé d'un capteur TDLAS naviguant dans
    un environnement contenant un panache de méthane. L'objectif est de
    détecter et localiser la source de fuite de manière efficace.
    
    Observation Space:
    - Position du drone (x, y, z)
    - Vitesse du drone (vx, vy, vz)
    - Mesure du capteur (concentration, détection)
    - Gradient local (gx, gy)
    - Temps écoulé (normalisé)
    
    Action Space:
    - Déplacement (Δx, Δy, Δz) normalisé [-1, 1]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, config: EnvironmentConfig, 
                 plume_config: Optional[PlumeConfig] = None,
                 sensor_config: Optional[TDLASConfig] = None):
        super().__init__()
        
        self.config = config
        self.world_width, self.world_height = config.world_size
        
        # Initialisation des composants
        if plume_config is None:
            plume_config = PlumeConfig()
        self.plume = MethanePlume(plume_config)
        
        if sensor_config is None:
            sensor_config = TDLASConfig()
        self.sensor = TDLASSensor(sensor_config)
        
        # État de l'environnement
        self.step_count = 0
        self.drone_position = np.array(config.initial_position + (config.initial_altitude,))
        self.drone_velocity = np.zeros(3)
        self.total_energy_consumed = 0.0
        self.detections = []
        self.trajectory = [self.drone_position.copy()]
        
        # Définition des espaces d'observation et d'action
        self._setup_spaces()
        
        # Historique pour la visualisation
        self.measurement_history = []
        self.reward_history = []
        
    def _setup_spaces(self):
        """Configure les espaces d'observation et d'action"""
        # Espace d'observation complet selon le MDP du document :
        # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, 
        #  concentration, detection, grad_x, grad_y,
        #  wind_x, wind_y, SNR, gp_prediction, gp_uncertainty, time]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, self.config.min_altitude, 
                         -self.config.max_speed, -self.config.max_speed, -self.config.max_speed,
                         0, 0, -1, -1, -1, -1, 0, 0, 0, 0]),
            high=np.array([self.world_width, self.world_height, self.config.max_altitude,
                          self.config.max_speed, self.config.max_speed, self.config.max_speed,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Espace d'action : déplacement normalisé [-1, 1] pour x, y, z
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Remet à zéro l'environnement"""
        super().reset(seed=seed)
        
        # Réinitialisation de l'état
        self.step_count = 0
        self.drone_position = np.array(self.config.initial_position + (self.config.initial_altitude,))
        self.drone_velocity = np.zeros(3)
        self.total_energy_consumed = 0.0
        self.detections = []
        self.trajectory = [self.drone_position.copy()]
        self.measurement_history = []
        self.reward_history = []
        
        # Réinitialisation des composants
        self.sensor.reset()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray, teacher: Optional[Any] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Exécute une étape de simulation
        
        Args:
            action: Action à exécuter (déplacement normalisé)
            teacher: Teacher (GaussianProcessTeacher) optionnel pour calculer la récompense éco-informative
            
        Returns:
            Tuple (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        
        # Conversion de l'action en déplacement
        displacement = self._action_to_displacement(action)
        
        # Mise à jour de la position
        old_position = self.drone_position.copy()
        self.drone_position += displacement
        
        # Application des contraintes
        self._apply_constraints()
        
        # Calcul de la vitesse
        self.drone_velocity = (self.drone_position - old_position) / self.config.time_step
        
        # Mesure de la concentration
        concentration = self.plume.concentration(
            self.drone_position[0], 
            self.drone_position[1], 
            self.step_count * self.config.time_step
        )
        
        # Mesure du capteur
        measured_conc, detected = self.sensor.measure_at_position(
            self.drone_position[0],
            self.drone_position[1], 
            self.drone_position[2],
            concentration
        )
        
        # Calcul du SNR (selon le document)
        snr = self.sensor.get_signal_to_noise_ratio(
            concentration,
            self.drone_position[2],  # altitude = distance
            surface_reflectivity=0.3
        )
        
        # Calcul du gradient local
        grad_x, grad_y = self.plume.gradient(
            self.drone_position[0],
            self.drone_position[1],
            self.step_count * self.config.time_step
        )
        
        # Enregistrement de la mesure
        self.measurement_history.append({
            'position': self.drone_position.copy(),
            'concentration': concentration,
            'measured_concentration': measured_conc,
            'detected': detected,
            'gradient': (grad_x, grad_y),
            'snr': snr
        })
        
        # Enregistrement des détections
        if detected:
            self.detections.append({
                'position': self.drone_position.copy(),
                'concentration': measured_conc,
                'step': self.step_count
            })
        
        # Mise à jour de la trajectoire
        self.trajectory.append(self.drone_position.copy())
        
        # Calcul de la récompense éco-informative (avec Teacher si disponible)
        reward = self._calculate_reward(action, concentration, detected, grad_x, grad_y, teacher=teacher)
        self.reward_history.append(reward)
        
        # Vérification des conditions de fin
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # Observation et info
        observation = self._get_observation(teacher)
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _action_to_displacement(self, action: np.ndarray) -> np.ndarray:
        """Convertit l'action normalisée en déplacement réel"""
        # Normalisation de l'action vers des déplacements en mètres
        max_displacement = self.config.max_speed * self.config.time_step
        
        displacement = action * max_displacement
        return displacement
    
    def _apply_constraints(self):
        """Applique les contraintes physiques du drone"""
        # Contraintes spatiales
        self.drone_position[0] = np.clip(self.drone_position[0], 0, self.world_width)
        self.drone_position[1] = np.clip(self.drone_position[1], 0, self.world_height)
        self.drone_position[2] = np.clip(self.drone_position[2], 
                                        self.config.min_altitude, 
                                        self.config.max_altitude)
    
    def _calculate_reward(self, action: np.ndarray, concentration: float, 
                         detected: bool, grad_x: float, grad_y: float,
                         teacher: Optional[Any] = None) -> float:
        """
        Calcule la récompense éco-informative selon le document :
        R(s,a) = α * ΔI(M_GP) - β * E(s,a)
        
        où :
        - ΔI(M_GP) : gain d'information (réduction d'incertitude du Teacher/GP)
        - E(s,a) : coût énergétique
        - α, β : poids respectifs
        """
        reward = 0.0
        
        # Gain d'information ΔI(M_GP) = réduction de l'incertitude
        information_gain = 0.0
        if teacher is not None and hasattr(teacher, 'gp'):
            # Calcul de l'incertitude avant et après la mesure
            # Approximation : réduction de l'incertitude locale
            x = self.drone_position[0]
            y = self.drone_position[1]
            
            try:
                # Prédiction du GP à la position actuelle
                X_pred = np.array([[x, y]])
                _, std = teacher.gp.predict(X_pred, return_std=True)
                uncertainty = std[0] if len(std) > 0 else 1.0
                
                # Gain d'information = réduction d'incertitude (inverse)
                information_gain = 1.0 / (1.0 + uncertainty)
            except:
                # Si le GP n'est pas encore entraîné, utiliser le gradient comme proxy
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                information_gain = gradient_magnitude
        else:
            # Si pas de Teacher, utiliser le gradient comme proxy du gain d'information
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            information_gain = gradient_magnitude
        
        # Coefficient α pour le gain d'information (selon document)
        alpha = 10.0  # Poids du gain d'information
        
        # Bonus d'information : α * ΔI(M_GP)
        reward += alpha * information_gain
        
        # Bonus pour détection (composante additionnelle)
        if detected:
            reward += self.config.detection_bonus
        
        # Pénalité énergétique : -β * E(s,a)
        energy_cost = self._calculate_energy_cost(action)
        beta = abs(self.config.energy_penalty)  # Coefficient β
        reward -= beta * energy_cost
        
        # Pénalité pour sortie des limites
        if (self.drone_position[0] <= 0 or self.drone_position[0] >= self.world_width or
            self.drone_position[1] <= 0 or self.drone_position[1] >= self.world_height):
            reward += self.config.boundary_penalty
        
        return reward
    
    def _calculate_energy_cost(self, action: np.ndarray) -> float:
        """
        Calcule le coût énergétique selon le modèle du document : 
        P ≈ c₁ * v_air³ + c₂ * |Δh / Δt|
        
        où :
        - v_air : vitesse par rapport à l'air (vitesse du drone + vent)
        - Δh / Δt : taux de changement d'altitude
        """
        # Vitesse du drone par rapport au sol
        speed_ground = np.linalg.norm(self.drone_velocity[:2])
        
        # Vitesse du vent (du modèle de panache)
        wind_vector = self.plume._wind_vector
        wind_speed = np.linalg.norm(wind_vector)
        
        # Vitesse par rapport à l'air (vecteur)
        air_velocity = self.drone_velocity[:2] - np.array(wind_vector)
        v_air = np.linalg.norm(air_velocity)
        
        # Puissance selon le modèle : P ≈ c₁ * v_air³ + c₂ * |Δh / Δt|
        # c₁ : coefficient pour la vitesse (dépendance cubique)
        c1 = self.config.speed_coefficient / (self.config.max_speed ** 2)  # Normalisation
        
        # c₂ : coefficient pour le changement d'altitude
        c2 = self.config.altitude_coefficient
        
        # Taux de changement d'altitude (Δh / Δt)
        dh_dt = abs(self.drone_velocity[2])  # Vitesse verticale
        
        # Calcul de la puissance selon le modèle du document
        # P ≈ c₁ * v_air³ + c₂ * |Δh / Δt|
        power = self.config.base_power + c1 * (v_air ** 3) + c2 * dh_dt
        
        # Énergie consommée
        energy = power * self.config.time_step
        self.total_energy_consumed += energy
        
        return energy
    
    def _is_terminated(self) -> bool:
        """Vérifie si l'épisode est terminé"""
        # L'épisode se termine si on détecte une concentration élevée
        if self.measurement_history:
            last_measurement = self.measurement_history[-1]
            if last_measurement['concentration'] > 0.5:  # Seuil élevé
                return True
        return False
    
    def _is_truncated(self) -> bool:
        """Vérifie si l'épisode est tronqué"""
        return self.step_count >= self.config.max_steps
    
    def _get_observation(self, teacher: Optional[Any] = None) -> np.ndarray:
        """
        Construit l'observation complète selon le MDP du document :
        s = (x, y, h, v, v_wind, SNR, M_GP)
        
        où :
        - (x, y, h) : position et altitude
        - v : vitesse du drone
        - v_wind : vecteur vent
        - SNR : rapport signal/bruit
        - M_GP : indicateurs du modèle GP (incertitude, prédiction)
        """
        # Position et vitesse
        pos = self.drone_position
        vel = self.drone_velocity
        
        # Vecteur vent (du modèle de panache)
        wind_vector = self.plume._wind_vector
        wind_speed = np.linalg.norm(wind_vector)
        wind_x = wind_vector[0] / 10.0 if len(wind_vector) > 0 else 0.0  # Normalisé
        wind_y = wind_vector[1] / 10.0 if len(wind_vector) > 1 else 0.0  # Normalisé
        
        # Mesure du capteur
        if self.measurement_history:
            last_measurement = self.measurement_history[-1]
            concentration = last_measurement['measured_concentration']
            detected = float(last_measurement['detected'])
            grad_x, grad_y = last_measurement['gradient']
            snr = last_measurement.get('snr', 0.0)
        else:
            concentration = 0.0
            detected = 0.0
            grad_x, grad_y = 0.0, 0.0
            snr = 0.0
        
        # Indicateurs du modèle GP (M_GP)
        gp_uncertainty = 1.0  # Par défaut : incertitude maximale
        gp_prediction = 0.0   # Par défaut : prédiction nulle
        
        if teacher is not None and hasattr(teacher, 'gp'):
            try:
                x = pos[0]
                y = pos[1]
                X_pred = np.array([[x, y]])
                
                # Prédiction et incertitude du GP
                mean, std = teacher.gp.predict(X_pred, return_std=True)
                gp_prediction = mean[0] if len(mean) > 0 else 0.0
                gp_uncertainty = std[0] if len(std) > 0 else 1.0
            except:
                # Si le GP n'est pas encore entraîné
                pass
        
        # Normalisation de l'incertitude GP
        gp_uncertainty_norm = np.clip(gp_uncertainty, 0.0, 1.0)
        gp_prediction_norm = np.clip(gp_prediction, 0.0, 1.0)
        
        # Normalisation du SNR (typiquement entre 0 et 100)
        snr_norm = np.clip(snr / 100.0, 0.0, 1.0)
        
        # Temps normalisé
        time_norm = self.step_count / self.config.max_steps
        
        # Construction de l'observation complète
        observation = np.array([
            pos[0] / self.world_width,           # Position X normalisée
            pos[1] / self.world_height,          # Position Y normalisée
            pos[2] / self.config.max_altitude,   # Position Z (altitude) normalisée
            vel[0] / self.config.max_speed,      # Vitesse X normalisée
            vel[1] / self.config.max_speed,      # Vitesse Y normalisée
            vel[2] / self.config.max_speed,      # Vitesse Z normalisée
            concentration,                       # Concentration mesurée
            detected,                           # Flag de détection
            grad_x,                            # Gradient X
            grad_y,                            # Gradient Y
            wind_x,                            # Composante X du vent normalisée
            wind_y,                            # Composante Y du vent normalisée
            snr_norm,                          # SNR normalisé
            gp_prediction_norm,                # Prédiction GP normalisée
            gp_uncertainty_norm,               # Incertitude GP normalisée
            time_norm                          # Temps normalisé
        ], dtype=np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Retourne les informations supplémentaires"""
        info = {
            'step': self.step_count,
            'position': self.drone_position.copy(),
            'total_energy': self.total_energy_consumed,
            'n_detections': len(self.detections),
            'trajectory_length': len(self.trajectory),
            'energy_cost': 0.0  # Initialisé à 0, sera mis à jour
        }
        
        if self.measurement_history:
            last_measurement = self.measurement_history[-1]
            info.update({
                'concentration': last_measurement['concentration'],
                'measured_concentration': last_measurement['measured_concentration'],
                'detected': last_measurement['detected'],
                'snr': last_measurement.get('snr', 0.0)
            })
        
        # Calcul du coût énergétique de la dernière action
        if len(self.measurement_history) > 0:
            # Le coût énergétique est déjà calculé dans _calculate_energy_cost
            # On le récupère depuis total_energy_consumed (sera mis à jour par la récompense)
            pass
        
        return info
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Rendu de l'environnement"""
        if mode == "human":
            self._render_human()
        elif mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Rendu pour affichage humain"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Carte du panache
        ax1 = axes[0, 0]
        self.plume.plot_plume(
            (0, self.world_width), 
            (0, self.world_height), 
            ax=ax1
        )
        
        # Trajectoire du drone
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax1.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=2, label='Trajectoire')
            ax1.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='o', 
                       label='Position actuelle')
        
        # Détections
        if self.detections:
            det_pos = np.array([d['position'] for d in self.detections])
            ax1.scatter(det_pos[:, 0], det_pos[:, 1], c='yellow', s=50, 
                       marker='*', label='Détections', zorder=5)
        
        ax1.set_title('Carte du Panache et Trajectoire')
        ax1.legend()
        
        # Historique des mesures
        ax2 = axes[0, 1]
        if self.measurement_history:
            steps = range(len(self.measurement_history))
            concentrations = [m['concentration'] for m in self.measurement_history]
            measured_conc = [m['measured_concentration'] for m in self.measurement_history]
            
            ax2.plot(steps, concentrations, 'b-', label='Concentration réelle')
            ax2.plot(steps, measured_conc, 'r--', label='Concentration mesurée')
            ax2.axhline(y=self.sensor.config.detection_threshold, color='orange', 
                       linestyle=':', label='Seuil détection')
        
        ax2.set_xlabel('Étape')
        ax2.set_ylabel('Concentration (kg/m³)')
        ax2.set_title('Historique des Mesures')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Historique des récompenses
        ax3 = axes[1, 0]
        if self.reward_history:
            ax3.plot(self.reward_history, 'g-', linewidth=2)
            ax3.set_xlabel('Étape')
            ax3.set_ylabel('Récompense')
            ax3.set_title('Historique des Récompenses')
            ax3.grid(True, alpha=0.3)
        
        # Métriques
        ax4 = axes[1, 1]
        metrics_text = f"""
        Étape: {self.step_count}
        Énergie totale: {self.total_energy_consumed:.1f} J
        Détections: {len(self.detections)}
        Position: ({self.drone_position[0]:.1f}, {self.drone_position[1]:.1f}, {self.drone_position[2]:.1f})
        """
        ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='center')
        ax4.set_title('Métriques')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _render_rgb_array(self) -> np.ndarray:
        """Rendu pour array RGB (pour enregistrement vidéo)"""
        # Implémentation simplifiée - retourne un array vide
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def close(self):
        """Ferme l'environnement"""
        pass


def create_test_environment() -> MethaneDetectionEnv:
    """Crée un environnement de test avec des paramètres réalistes"""
    env_config = EnvironmentConfig(
        world_size=(100.0, 100.0),
        max_steps=500,
        initial_position=(10.0, 10.0),
        initial_altitude=5.0
    )
    
    plume_config = PlumeConfig(
        leak_x=60.0,
        leak_y=60.0,
        leak_intensity=0.3,
        wind_speed=2.0,
        wind_direction=45
    )
    
    sensor_config = TDLASConfig(
        noise_level=0.05,
        detection_threshold=0.02
    )
    
    return MethaneDetectionEnv(env_config, plume_config, sensor_config)


if __name__ == "__main__":
    # Test de l'environnement
    env = create_test_environment()
    
    print("Test de l'environnement de simulation:")
    print("=" * 50)
    
    # Reset
    obs, info = env.reset()
    print(f"Observation initiale: {obs}")
    print(f"Info: {info}")
    
    # Simulation de quelques étapes
    for i in range(10):
        # Action aléatoire
        action = env.action_space.sample()
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Étape {i+1}: Action {action}, Reward {reward:.3f}, "
              f"Position {info['position']}")
        
        if terminated or truncated:
            break
    
    # Rendu
    env.render()
    
    # Fermeture
    env.close()




