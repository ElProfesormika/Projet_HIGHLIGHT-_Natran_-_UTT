"""
Module de simulation pour HIGHLIGHT+
Contient les modèles physiques et l'environnement de simulation
"""

from .plume_model import MethanePlume, PlumeConfig
from .environment import MethaneDetectionEnv, EnvironmentConfig

__all__ = [
    'MethanePlume', 'PlumeConfig',
    'MethaneDetectionEnv', 'EnvironmentConfig'
]
















