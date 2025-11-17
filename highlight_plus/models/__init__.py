"""
Module de modèles IA pour HIGHLIGHT+
Contient l'Expert (Teacher) et l'Apprenti (Student)
"""

from .teacher_gp import GaussianProcessTeacher, TeacherConfig
from .student_rl import StudentRL, StudentConfig

__all__ = [
    'GaussianProcessTeacher', 'TeacherConfig',
    'StudentRL', 'StudentConfig'
]










