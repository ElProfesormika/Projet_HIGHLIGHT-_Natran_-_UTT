"""
Chargeur de configuration pour HIGHLIGHT+
Gestion centralisée des paramètres de configuration
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """Configuration complète du système HIGHLIGHT+"""
    physics: Dict[str, Any]
    sensor: Dict[str, Any]
    drone: Dict[str, Any]
    simulation: Dict[str, Any]
    teacher: Dict[str, Any]
    student: Dict[str, Any]
    reward: Dict[str, Any]
    experiment: Dict[str, Any]


class ConfigLoader:
    """Chargeur de configuration pour HIGHLIGHT+"""
    
    @staticmethod
    def load_config(config_path: str = "configs/default.yaml") -> Config:
        """
        Charge la configuration depuis un fichier YAML
        
        Args:
            config_path: Chemin vers le fichier de configuration
            
        Returns:
            Objet Config avec tous les paramètres
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return Config(**config_data)
    
    @staticmethod
    def save_config(config: Config, config_path: str):
        """
        Sauvegarde la configuration dans un fichier YAML
        
        Args:
            config: Objet Config à sauvegarder
            config_path: Chemin de sauvegarde
        """
        config_dict = asdict(config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def create_default_config() -> Config:
        """Crée une configuration par défaut"""
        return Config(
            physics={
                'leak_source': {
                    'x': 50.0,
                    'y': 50.0,
                    'intensity': 1.0
                },
                'environment': {
                    'wind_speed': 2.0,
                    'wind_direction': 45,
                    'temperature': 288.15,
                    'pressure': 101325,
                    'humidity': 0.6
                },
                'diffusion': {
                    'sigma_x': 5.0,
                    'sigma_y': 3.0,
                    'decay_rate': 0.01
                }
            },
            sensor={
                'noise_level': 0.1,
                'detection_threshold': 0.05,
                'range_max': 100.0,
                'range_min': 1.0,
                'update_frequency': 10.0,
                'measurement_time': 0.1,
                'laser_wavelength': 1653.7,
                'beam_divergence': 0.1,
                'power': 10.0,
                'atmospheric_noise': 0.05,
                'electronic_noise': 0.02,
                'interference_factor': 0.1
            },
            drone={
                'max_speed': 5.0,
                'max_altitude': 20.0,
                'min_altitude': 2.0,
                'energy': {
                    'base_power': 100.0,
                    'speed_coefficient': 50.0,
                    'altitude_coefficient': 25.0
                }
            },
            simulation={
                'world_size': [100, 100],
                'time_step': 0.1,
                'max_steps': 1000,
                'initial_position': [10, 10],
                'initial_altitude': 5.0
            },
            teacher={
                'kernel': {
                    'type': 'RBF',
                    'length_scale': 10.0,
                    'variance': 1.0
                },
                'alpha': 1e-3,
                'n_restarts': 10,
                'exploration': {
                    'acquisition_function': 'UCB',
                    'beta': 2.0
                }
            },
            student={
                'algorithm': 'PPO',
                'network': {
                    'hidden_layers': [256, 256, 128],
                    'activation': 'tanh'
                },
                'training': {
                    'total_timesteps': 100000,
                    'learning_rate': 3e-4,
                    'batch_size': 64,
                    'n_epochs': 10
                },
                'distillation': {
                    'lambda_kl': 0.1,
                    'temperature': 3.0
                }
            },
            reward={
                'alpha': 1.0,
                'beta': 0.5,
                'gamma': 0.99,
                'detection_bonus': 10.0,
                'energy_penalty': -0.1,
                'boundary_penalty': -5.0
            },
            experiment={
                'n_runs': 10,
                'save_trajectories': True,
                'save_metrics': True,
                'plot_frequency': 100,
                'save_plots': True,
                'log_level': 'INFO',
                'tensorboard': True
            }
        )










