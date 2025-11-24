"""
HIGHLIGHT+ - Version Simplifiée pour Démonstration
Simulateur 2D avec modèle gaussien simple
"""

from .simple_simulator import (
    SimpleConfig,
    SimpleSimulator,
    SimpleMethanePlume,
    SimpleDrone,
    NaiveAgent,
    HighlightAgent
)

from .comparative_analysis import ComparativeAnalyzer

__version__ = "1.0.0"
__all__ = [
    'SimpleConfig',
    'SimpleSimulator',
    'SimpleMethanePlume',
    'SimpleDrone',
    'NaiveAgent',
    'HighlightAgent',
    'ComparativeAnalyzer'
]










