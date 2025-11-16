"""
Expériences comparatives pour HIGHLIGHT+
Comparaison Teacher vs Student vs approches baselines
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import yaml
import json
import time
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

# Import des composants du projet
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import MethaneDetectionEnv, EnvironmentConfig
from simulation.plume_model import MethanePlume, PlumeConfig
from sensors.tdlas_sensor import TDLASSensor, TDLASConfig
from models.teacher_gp import GaussianProcessTeacher, TeacherConfig
from models.student_rl import StudentRL, StudentConfig
from visualization.plotter import HighlightPlotter, PlotConfig


@dataclass
class ExperimentConfig:
    """Configuration des expériences"""
    # Paramètres généraux
    n_runs: int = 10
    max_steps: int = 500
    save_results: bool = True
    save_plots: bool = True
    output_dir: str = "results"
    
    # Paramètres de l'environnement
    world_size: Tuple[float, float] = (100.0, 100.0)
    leak_position: Tuple[float, float] = (60.0, 60.0)
    leak_intensity: float = 0.3
    
    # Paramètres des agents
    teacher_config: TeacherConfig = None
    student_config: StudentConfig = None


class ExperimentRunner:
    """
    Gestionnaire d'expériences pour HIGHLIGHT+
    
    Exécute des expériences comparatives entre :
    1. Expert (Teacher) - Processus Gaussiens
    2. Apprenti (Student) - RL avec distillation
    3. Baseline - Trajectoire aléatoire
    4. Baseline - Trajectoire en spirale
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.plotter = HighlightPlotter()
        
        # Création du répertoire de sortie
        Path(config.output_dir).mkdir(exist_ok=True)
        
    def setup_environment(self) -> MethaneDetectionEnv:
        """Configure l'environnement de simulation"""
        env_config = EnvironmentConfig(
            world_size=self.config.world_size,
            max_steps=self.config.max_steps,
            initial_position=(10.0, 10.0),
            initial_altitude=5.0
        )
        
        plume_config = PlumeConfig(
            leak_x=self.config.leak_position[0],
            leak_y=self.config.leak_position[1],
            leak_intensity=self.config.leak_intensity,
            wind_speed=2.0,
            wind_direction=45
        )
        
        sensor_config = TDLASConfig(
            noise_level=0.05,
            detection_threshold=0.02
        )
        
        return MethaneDetectionEnv(env_config, plume_config, sensor_config)
    
    def run_teacher_experiment(self, env: MethaneDetectionEnv) -> Dict[str, Any]:
        """Exécute l'expérience avec l'Expert (Teacher)"""
        print("Exécution de l'expérience Teacher...")
        
        # Configuration du Teacher
        teacher_config = self.config.teacher_config or TeacherConfig(
            kernel_length_scale=8.0,
            exploration_parameter=2.5,
            max_step_size=8.0
        )
        
        world_bounds = (0, self.config.world_size[0], 0, self.config.world_size[1])
        teacher = GaussianProcessTeacher(teacher_config, world_bounds)
        
        # Reset de l'environnement
        obs, info = env.reset()
        current_pos = info['position']
        
        # Initialisation
        teacher.current_position = (current_pos[0], current_pos[1])
        
        # Simulation
        trajectory = [current_pos[:2]]
        detections = []
        energy_consumption = []
        rewards = []
        
        for step in range(self.config.max_steps):
            # Mesure de la concentration
            concentration = env.plume.concentration(current_pos[0], current_pos[1], step * 0.1)
            measured_conc, detected = env.sensor.measure_at_position(
                current_pos[0], current_pos[1], current_pos[2], concentration
            )
            
            # Ajout de l'observation au Teacher
            teacher.add_observation(current_pos[0], current_pos[1], measured_conc)
            
            # Calcul du gradient pour guider la navigation
            grad_x, grad_y = env.plume.gradient(
                current_pos[0],
                current_pos[1],
                step * 0.1
            )
            
            # Sélection du prochain point avec gradient
            next_x, next_y = teacher.select_next_point(
                current_pos[0], 
                current_pos[1],
                gradient_x=grad_x,
                gradient_y=grad_y
            )
            
            # Calcul de l'action avec combinaison Teacher + Gradient
            direction = np.array([next_x - current_pos[0], next_y - current_pos[1]])
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0.1:
                # Normaliser la direction
                direction = direction / direction_norm
                
                # Combiner avec gradient pour robustesse
                grad_norm = np.linalg.norm([grad_x, grad_y])
                if grad_norm > 1e-6:
                    gradient_direction = np.array([grad_x, grad_y]) / grad_norm
                    combined_direction = 0.7 * direction + 0.3 * gradient_direction
                    combined_norm = np.linalg.norm(combined_direction)
                    if combined_norm > 1e-6:
                        combined_direction = combined_direction / combined_norm
                    else:
                        combined_direction = direction
                else:
                    combined_direction = direction
                
                action = np.array([
                    combined_direction[0] * 0.8,
                    combined_direction[1] * 0.8,
                    0.0
                ])
            else:
                # Si direction trop petite, utiliser le gradient
                grad_norm = np.linalg.norm([grad_x, grad_y])
                if grad_norm > 1e-6:
                    action = np.array([
                        grad_x / grad_norm * 0.5,
                        grad_y / grad_norm * 0.5,
                        0.0
                    ])
                else:
                    action = np.array([0.0, 0.0, 0.0])
            
            action = np.clip(action, -1, 1)
            
            # Step de l'environnement
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Mise à jour
            current_pos = info['position']
            trajectory.append(current_pos[:2])
            rewards.append(reward)
            energy_consumption.append(info['total_energy'])
            
            if detected:
                detections.append({
                    'position': current_pos.copy(),
                    'concentration': measured_conc,
                    'step': step
                })
            
            if terminated or truncated:
                break
        
        # Métriques de performance
        metrics = self._calculate_metrics(trajectory, detections, energy_consumption, rewards)
        
        return {
            'trajectory': trajectory,
            'detections': detections,
            'energy_consumption': energy_consumption,
            'rewards': rewards,
            'metrics': metrics,
            'teacher': teacher
        }
    
    def run_student_experiment(self, env: MethaneDetectionEnv) -> Dict[str, Any]:
        """Exécute l'expérience avec l'Apprenti (Student)"""
        print("Exécution de l'expérience Student...")
        
        # Configuration du Student
        student_config = self.config.student_config or StudentConfig(
            hidden_layers=[128, 64],
            learning_rate=1e-3,
            batch_size=32,
            lambda_kl=0.2
        )
        
        # Création du Student (sans Teacher pour simplifier)
        student = StudentRL(11, 3, student_config)  # 11 obs, 3 actions
        
        # Reset de l'environnement
        obs, info = env.reset()
        
        # Simulation
        trajectory = [info['position'][:2]]
        detections = []
        energy_consumption = []
        rewards = []
        
        for step in range(self.config.max_steps):
            # Obtenir le gradient pour guider l'action
            current_pos = info.get('position', obs[:3])
            grad_x, grad_y = env.plume.gradient(
                current_pos[0],
                current_pos[1],
                step * 0.1
            )
            
            # Sélection de l'action
            action = student.select_action(obs, training=True)
            
            # Améliorer l'action avec le gradient pour plus de robustesse
            grad_norm = np.linalg.norm([grad_x, grad_y])
            if grad_norm > 1e-6:
                # Combiner action Student (80%) avec gradient (20%) pour guider l'apprentissage
                gradient_direction = np.array([grad_x / grad_norm, grad_y / grad_norm, 0.0])
                action = 0.8 * action + 0.2 * gradient_direction
                action = np.clip(action, -1, 1)
            
            # Step de l'environnement
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Stockage de l'expérience
            student.store_experience(obs, action, reward, next_obs, terminated or truncated)
            
            # Apprentissage
            if len(student.replay_buffer) > student.config.learning_starts:
                student.learn()
            
            # Mise à jour
            obs = next_obs
            trajectory.append(info['position'][:2])
            rewards.append(reward)
            energy_consumption.append(info['total_energy'])
            
            if info.get('detected', False):
                detections.append({
                    'position': info['position'].copy(),
                    'concentration': info.get('measured_concentration', 0),
                    'step': step
                })
            
            if terminated or truncated:
                break
        
        # Métriques de performance
        metrics = self._calculate_metrics(trajectory, detections, energy_consumption, rewards)
        
        return {
            'trajectory': trajectory,
            'detections': detections,
            'energy_consumption': energy_consumption,
            'rewards': rewards,
            'metrics': metrics,
            'student': student
        }
    
    def run_random_baseline(self, env: MethaneDetectionEnv) -> Dict[str, Any]:
        """Exécute l'expérience baseline avec trajectoire aléatoire"""
        print("Exécution de l'expérience Random Baseline...")
        
        # Reset de l'environnement
        obs, info = env.reset()
        
        # Simulation
        trajectory = [info['position'][:2]]
        detections = []
        energy_consumption = []
        rewards = []
        
        for step in range(self.config.max_steps):
            # Action aléatoire
            action = env.action_space.sample()
            
            # Step de l'environnement
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Mise à jour
            trajectory.append(info['position'][:2])
            rewards.append(reward)
            energy_consumption.append(info['total_energy'])
            
            if info.get('detected', False):
                detections.append({
                    'position': info['position'].copy(),
                    'concentration': info.get('measured_concentration', 0),
                    'step': step
                })
            
            if terminated or truncated:
                break
        
        # Métriques de performance
        metrics = self._calculate_metrics(trajectory, detections, energy_consumption, rewards)
        
        return {
            'trajectory': trajectory,
            'detections': detections,
            'energy_consumption': energy_consumption,
            'rewards': rewards,
            'metrics': metrics
        }
    
    def run_spiral_baseline(self, env: MethaneDetectionEnv) -> Dict[str, Any]:
        """Exécute l'expérience baseline avec trajectoire en spirale"""
        print("Exécution de l'expérience Spiral Baseline...")
        
        # Reset de l'environnement
        obs, info = env.reset()
        initial_pos = info['position']
        
        # Simulation
        trajectory = [initial_pos[:2]]
        detections = []
        energy_consumption = []
        rewards = []
        
        # Paramètres de la spirale
        center_x, center_y = self.config.world_size[0] / 2, self.config.world_size[1] / 2
        radius = 5.0
        angle_step = 0.1
        
        for step in range(self.config.max_steps):
            # Calcul de la position en spirale
            angle = step * angle_step
            target_x = center_x + radius * np.cos(angle)
            target_y = center_y + radius * np.sin(angle)
            
            # Augmentation progressive du rayon
            radius += 0.05
            
            # Calcul de l'action
            current_pos = info['position']
            action = np.array([
                (target_x - current_pos[0]) / 5.0,
                (target_y - current_pos[1]) / 5.0,
                0.0
            ])
            action = np.clip(action, -1, 1)
            
            # Step de l'environnement
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Mise à jour
            trajectory.append(info['position'][:2])
            rewards.append(reward)
            energy_consumption.append(info['total_energy'])
            
            if info.get('detected', False):
                detections.append({
                    'position': info['position'].copy(),
                    'concentration': info.get('measured_concentration', 0),
                    'step': step
                })
            
            if terminated or truncated:
                break
        
        # Métriques de performance
        metrics = self._calculate_metrics(trajectory, detections, energy_consumption, rewards)
        
        return {
            'trajectory': trajectory,
            'detections': detections,
            'energy_consumption': energy_consumption,
            'rewards': rewards,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, trajectory: List[Tuple[float, float]], 
                          detections: List[Dict], energy_consumption: List[float],
                          rewards: List[float]) -> Dict[str, float]:
        """Calcule les métriques de performance"""
        # Taux de détection
        detection_rate = len(detections) / len(trajectory) if trajectory else 0.0
        
        # Distance totale parcourue
        total_distance = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        # Efficacité énergétique
        total_energy = sum(energy_consumption)
        energy_efficiency = total_distance / total_energy if total_energy > 0 else 0.0
        
        # Précision de localisation (distance à la source réelle)
        if detections:
            leak_x, leak_y = self.config.leak_position
            distances = []
            for detection in detections:
                pos = detection['position']
                dist = np.sqrt((pos[0] - leak_x)**2 + (pos[1] - leak_y)**2)
                distances.append(dist)
            localization_accuracy = 1.0 / (1.0 + np.mean(distances))  # Plus proche = plus précis
        else:
            localization_accuracy = 0.0
        
        # Récompense totale
        total_reward = sum(rewards)
        
        # Temps de mission
        mission_time = len(trajectory)
        
        return {
            'detection_rate': detection_rate,
            'total_distance': total_distance,
            'energy_efficiency': energy_efficiency,
            'localization_accuracy': localization_accuracy,
            'total_reward': total_reward,
            'mission_time': mission_time,
            'total_energy': total_energy
        }
    
    def run_comparison(self) -> Dict[str, Any]:
        """Exécute la comparaison complète"""
        print("Démarrage de la comparaison HIGHLIGHT+...")
        print("=" * 50)
        
        # Configuration de l'environnement
        env = self.setup_environment()
        
        # Exécution des expériences
        results = {}
        
        # Teacher
        results['teacher'] = self.run_teacher_experiment(env)
        
        # Student
        results['student'] = self.run_student_experiment(env)
        
        # Baselines
        results['random'] = self.run_random_baseline(env)
        results['spiral'] = self.run_spiral_baseline(env)
        
        # Sauvegarde des résultats
        if self.config.save_results:
            self._save_results(results)
        
        # Génération des visualisations
        if self.config.save_plots:
            self._generate_plots(results)
        
        # Affichage des résultats
        self._print_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Sauvegarde les résultats"""
        output_file = Path(self.config.output_dir) / "comparison_results.json"
        
        # Conversion des résultats en format JSON-serializable
        json_results = {}
        for method, data in results.items():
            json_results[method] = {
                'metrics': data['metrics'],
                'trajectory': data['trajectory'],
                'detections': data['detections'],
                'energy_consumption': data['energy_consumption'],
                'rewards': data['rewards']
            }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Résultats sauvegardés dans {output_file}")
    
    def _generate_plots(self, results: Dict[str, Any]):
        """Génère les graphiques de comparaison"""
        # Comparaison des trajectoires
        fig = self.plotter.plot_trajectory_comparison(
            teacher_trajectory=results['teacher']['trajectory'],
            student_trajectory=results['student']['trajectory'],
            detections=results['teacher']['detections'],
            title="Comparaison des Trajectoires - HIGHLIGHT+"
        )
        plt.savefig(Path(self.config.output_dir) / "trajectory_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Comparaison A/B
        fig = self.plotter.plot_ab_comparison(
            results_a=results['teacher']['metrics'],
            results_b=results['student']['metrics'],
            title="Comparaison A/B - Teacher vs Student"
        )
        plt.savefig(Path(self.config.output_dir) / "ab_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Comparaison de la consommation énergétique
        fig = self.plotter.plot_energy_consumption(
            teacher_energy=results['teacher']['energy_consumption'],
            student_energy=results['student']['energy_consumption'],
            title="Consommation Énergétique - Teacher vs Student"
        )
        plt.savefig(Path(self.config.output_dir) / "energy_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graphiques sauvegardés dans {self.config.output_dir}")
    
    def _print_results(self, results: Dict[str, Any]):
        """Affiche les résultats de la comparaison"""
        print("\n" + "=" * 60)
        print("RÉSULTATS DE LA COMPARAISON HIGHLIGHT+")
        print("=" * 60)
        
        methods = ['teacher', 'student', 'random', 'spiral']
        method_names = ['Expert (Teacher)', 'Apprenti (Student)', 'Random Baseline', 'Spiral Baseline']
        
        # Tableau des métriques
        metrics_table = []
        for method, name in zip(methods, method_names):
            metrics = results[method]['metrics']
            metrics_table.append([
                name,
                f"{metrics['detection_rate']:.3f}",
                f"{metrics['energy_efficiency']:.3f}",
                f"{metrics['localization_accuracy']:.3f}",
                f"{metrics['total_energy']:.1f}",
                f"{metrics['mission_time']}"
            ])
        
        # Affichage du tableau
        df = pd.DataFrame(metrics_table, columns=[
            'Méthode', 'Taux Détection', 'Efficacité Énergétique', 
            'Précision Localisation', 'Énergie Totale (J)', 'Temps Mission'
        ])
        
        print("\nTableau des Métriques:")
        print(df.to_string(index=False))
        
        # Analyse comparative
        print("\n" + "=" * 60)
        print("ANALYSE COMPARATIVE")
        print("=" * 60)
        
        teacher_metrics = results['teacher']['metrics']
        student_metrics = results['student']['metrics']
        
        # Améliorations du Student par rapport au Teacher
        detection_improvement = (student_metrics['detection_rate'] - teacher_metrics['detection_rate']) / teacher_metrics['detection_rate'] * 100
        energy_improvement = (student_metrics['energy_efficiency'] - teacher_metrics['energy_efficiency']) / teacher_metrics['energy_efficiency'] * 100
        
        print(f"Amélioration du taux de détection: {detection_improvement:+.1f}%")
        print(f"Amélioration de l'efficacité énergétique: {energy_improvement:+.1f}%")
        
        # Comparaison avec les baselines
        random_metrics = results['random']['metrics']
        spiral_metrics = results['spiral']['metrics']
        
        print(f"\nAvantage du Student vs Random:")
        print(f"  - Taux de détection: {student_metrics['detection_rate'] / random_metrics['detection_rate']:.1f}x")
        print(f"  - Efficacité énergétique: {student_metrics['energy_efficiency'] / random_metrics['energy_efficiency']:.1f}x")
        
        print(f"\nAvantage du Student vs Spiral:")
        print(f"  - Taux de détection: {student_metrics['detection_rate'] / spiral_metrics['detection_rate']:.1f}x")
        print(f"  - Efficacité énergétique: {student_metrics['energy_efficiency'] / spiral_metrics['energy_efficiency']:.1f}x")


def main():
    """Fonction principale pour exécuter les expériences"""
    # Configuration des expériences
    config = ExperimentConfig(
        n_runs=5,
        max_steps=300,
        save_results=True,
        save_plots=True,
        output_dir="results"
    )
    
    # Exécution des expériences
    runner = ExperimentRunner(config)
    results = runner.run_comparison()
    
    print("\nExpériences terminées avec succès!")
    print(f"Résultats disponibles dans le dossier: {config.output_dir}")


if __name__ == "__main__":
    main()






