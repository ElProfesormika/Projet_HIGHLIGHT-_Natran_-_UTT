"""
HIGHLIGHT+ - Système de détection intelligente de micro-fuites de méthane
Concours Innovation Natran x Fondation UTT

Ce package implémente une architecture Teacher-Student pour l'optimisation
des trajectoires de vol de drones dans la détection de fuites de méthane.

Architecture:
- Expert (Teacher): Processus Gaussiens pour l'apprentissage actif
- Apprenti (Student): Apprentissage par renforcement avec distillation
- Simulation: Modèle physique du panache + capteur TDLAS
- Visualisation: Outils d'analyse et de comparaison

Auteurs:
- Housséni YABRE (Lead AI Engineer)
- Kabinet SYLLA (Simulation & Physics)
- Nobert Bassooma DIDANERA (System Integration)
"""

__version__ = "1.0.0"
__author__ = "Équipe HIGHLIGHT+"
__email__ = "highlight.plus@utt.fr"

# Import des composants principaux
from .simulation.plume_model import MethanePlume, PlumeConfig
from .sensors.tdlas_sensor import TDLASSensor, TDLASConfig
from .models.teacher_gp import GaussianProcessTeacher, TeacherConfig
from .models.student_rl import StudentRL, StudentConfig
from .simulation.environment import MethaneDetectionEnv, EnvironmentConfig
from .visualization.plotter import HighlightPlotter, PlotConfig

__all__ = [
    # Modèles
    'MethanePlume', 'PlumeConfig',
    'TDLASSensor', 'TDLASConfig',
    'GaussianProcessTeacher', 'TeacherConfig',
    'StudentRL', 'StudentConfig',
    'MethaneDetectionEnv', 'EnvironmentConfig',
    'HighlightPlotter', 'PlotConfig',
    
    # Métadonnées
    '__version__', '__author__', '__email__'
]









