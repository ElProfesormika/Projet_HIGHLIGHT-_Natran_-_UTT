"""
Module d'analyse pour HIGHLIGHT+
Contient les outils d'analyse de l'apprentissage et des performances
"""

from .learning_analysis import LearningAnalyzer, LearningMetrics
from .enhanced_detector import EnhancedDetector, DetectionEvent
from .performance_validator import PerformanceValidator, PerformanceMetrics, LocalizationAccuracy
from .methane_leak_validator import MethaneLeakValidator

__all__ = [
    'LearningAnalyzer', 
    'LearningMetrics',
    'EnhancedDetector',
    'DetectionEvent',
    'PerformanceValidator',
    'PerformanceMetrics',
    'LocalizationAccuracy',
    'MethaneLeakValidator'
]









