"""
Test de détection sur différentes positions de fuites
Évalue la robustesse du système HIGHLIGHT+ sur diverses configurations
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
from dataclasses import dataclass
import json
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
from analysis.learning_analysis import LearningAnalyzer, LearningMetrics


@dataclass
class LeakTestConfig:
    """Configuration pour les tests de fuites"""
    # Positions de fuites à tester
    leak_positions: List[Tuple[float, float]]
    
    # Paramètres de test
    n_runs_per_position: int = 5
    max_steps: int = 500
    
    # Paramètres de l'environnement
    world_size: Tuple[float, float] = (100.0, 100.0)
    initial_position: Tuple[float, float] = (10.0, 10.0)
    
    # Paramètres du panache
    leak_intensity: float = 0.3
    wind_speed: float = 2.0
    wind_direction: float = 45.0


class LeakPositionTester:
    """
    Testeur de positions de fuites pour HIGHLIGHT+
    
    Évalue :
    1. Capacité de détection sur différentes positions
    2. Robustesse du système
    3. Temps de détection selon la position
    4. Efficacité énergétique
    """
    
    def __init__(self, config: LeakTestConfig):
        self.config = config
        self.results = {}
        self.learning_analyzers = {}
        
    def run_single_test(self, leak_position: Tuple[float, float], 
                       run_id: int) -> Dict[str, Any]:
        """
        Exécute un test sur une position de fuite donnée
        
        Args:
            leak_position: Position de la fuite (x, y)
            run_id: Identifiant du run
            
        Returns:
            Dictionnaire avec les résultats du test
        """
        print(f"Test position {leak_position}, run {run_id}")
        
        # Configuration de l'environnement
        env_config = EnvironmentConfig(
            world_size=self.config.world_size,
            max_steps=self.config.max_steps,
            initial_position=self.config.initial_position,
            initial_altitude=5.0
        )
        
        # Configuration du panache
        plume_config = PlumeConfig(
            leak_x=leak_position[0],
            leak_y=leak_position[1],
            leak_intensity=self.config.leak_intensity,
            wind_speed=self.config.wind_speed,
            wind_direction=self.config.wind_direction
        )
        
        # Configuration du capteur
        sensor_config = TDLASConfig(
            noise_level=0.05,
            detection_threshold=0.02
        )
        
        # Création de l'environnement
        env = MethaneDetectionEnv(env_config, plume_config, sensor_config)
        
        # Configuration du Teacher
        teacher_config = TeacherConfig(
            kernel_length_scale=8.0,
            exploration_parameter=2.5,
            max_step_size=8.0
        )
        
        world_bounds = (0, self.config.world_size[0], 0, self.config.world_size[1])
        teacher = GaussianProcessTeacher(teacher_config, world_bounds)
        
        # Configuration du Student
        student_config = StudentConfig(
            hidden_layers=[128, 64],
            learning_rate=1e-3,
            batch_size=32,
            lambda_kl=0.2
        )
        
        student = StudentRL(11, 3, student_config)
        
        # Analyseur d'apprentissage
        analyzer = LearningAnalyzer()
        
        # Simulation
        obs, info = env.reset()
        current_pos = info['position']
        teacher.current_position = (current_pos[0], current_pos[1])
        
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
            
            # Sélection de l'action (Teacher pour la démonstration)
            next_x, next_y = teacher.select_next_point(current_pos[0], current_pos[1])
            
            # Calcul de l'action
            action = np.array([
                (next_x - current_pos[0]) / 5.0,
                (next_y - current_pos[1]) / 5.0,
                0.0
            ])
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
            
            # Ajout des métriques d'apprentissage (simulation)
            learning_metrics = LearningMetrics(
                step=step,
                loss_rl=0.1 * np.exp(-step/100),  # Simulation
                loss_kl=0.05 * np.exp(-step/150),  # Simulation
                total_loss=0.1 * np.exp(-step/100) + 0.05 * np.exp(-step/150),
                epsilon=max(0.01, 1.0 - step/200),
                reward=reward,
                detection=detected,
                concentration=measured_conc,
                position=(current_pos[0], current_pos[1])
            )
            analyzer.add_learning_step(learning_metrics)
            
            if terminated or truncated:
                break
        
        # Calcul des métriques
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
        
        # Précision de localisation
        if detections:
            distances_to_leak = []
            for detection in detections:
                pos = detection['position']
                dist = np.sqrt((pos[0] - leak_position[0])**2 + (pos[1] - leak_position[1])**2)
                distances_to_leak.append(dist)
            localization_accuracy = 1.0 / (1.0 + np.mean(distances_to_leak))
        else:
            localization_accuracy = 0.0
        
        # Temps de première détection
        first_detection_step = detections[0]['step'] if detections else self.config.max_steps
        
        # Analyse de l'apprentissage
        convergence_analysis = analyzer.analyze_learning_convergence()
        detection_analysis = analyzer.analyze_detection_capability()
        
        results = {
            'leak_position': leak_position,
            'run_id': run_id,
            'detection_rate': detection_rate,
            'total_distance': total_distance,
            'energy_efficiency': energy_efficiency,
            'localization_accuracy': localization_accuracy,
            'first_detection_step': first_detection_step,
            'total_energy': total_energy,
            'n_detections': len(detections),
            'trajectory_length': len(trajectory),
            'convergence_analysis': convergence_analysis,
            'detection_analysis': detection_analysis,
            'trajectory': trajectory,
            'detections': detections
        }
        
        env.close()
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Exécute tous les tests sur toutes les positions
        
        Returns:
            Dictionnaire avec tous les résultats
        """
        print("Démarrage des tests de positions de fuites...")
        print("=" * 50)
        
        all_results = {}
        
        for leak_pos in self.config.leak_positions:
            print(f"\nTest de la position de fuite: {leak_pos}")
            position_results = []
            
            for run_id in range(self.config.n_runs_per_position):
                try:
                    result = self.run_single_test(leak_pos, run_id)
                    position_results.append(result)
                    
                    print(f"  Run {run_id+1}: {result['n_detections']} détections, "
                          f"étape {result['first_detection_step']}, "
                          f"efficacité {result['energy_efficiency']:.3f}")
                    
                except Exception as e:
                    print(f"  Erreur run {run_id+1}: {e}")
                    continue
            
            all_results[str(leak_pos)] = position_results
        
        # Analyse globale
        global_analysis = self._analyze_global_results(all_results)
        all_results['global_analysis'] = global_analysis
        
        return all_results
    
    def _analyze_global_results(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyse les résultats globaux"""
        global_analysis = {}
        
        # Statistiques par position
        position_stats = {}
        for pos_str, position_results in results.items():
            if pos_str == 'global_analysis':
                continue
                
            if not position_results:
                continue
            
            # Calcul des moyennes
            detection_rates = [r['detection_rate'] for r in position_results]
            energy_efficiencies = [r['energy_efficiency'] for r in position_results]
            localization_accuracies = [r['localization_accuracy'] for r in position_results]
            first_detection_steps = [r['first_detection_step'] for r in position_results]
            
            position_stats[pos_str] = {
                'mean_detection_rate': np.mean(detection_rates),
                'std_detection_rate': np.std(detection_rates),
                'mean_energy_efficiency': np.mean(energy_efficiencies),
                'std_energy_efficiency': np.std(energy_efficiencies),
                'mean_localization_accuracy': np.mean(localization_accuracies),
                'std_localization_accuracy': np.std(localization_accuracies),
                'mean_first_detection_step': np.mean(first_detection_steps),
                'std_first_detection_step': np.std(first_detection_steps),
                'n_successful_runs': len(position_results)
            }
        
        global_analysis['position_stats'] = position_stats
        
        # Statistiques globales
        all_detection_rates = []
        all_energy_efficiencies = []
        all_localization_accuracies = []
        
        for pos_str, position_results in results.items():
            if pos_str == 'global_analysis':
                continue
            for result in position_results:
                all_detection_rates.append(result['detection_rate'])
                all_energy_efficiencies.append(result['energy_efficiency'])
                all_localization_accuracies.append(result['localization_accuracy'])
        
        global_analysis['overall_stats'] = {
            'mean_detection_rate': np.mean(all_detection_rates),
            'std_detection_rate': np.std(all_detection_rates),
            'mean_energy_efficiency': np.mean(all_energy_efficiencies),
            'std_energy_efficiency': np.std(all_energy_efficiencies),
            'mean_localization_accuracy': np.mean(all_localization_accuracies),
            'std_localization_accuracy': np.std(all_localization_accuracies),
            'total_tests': len(all_detection_rates)
        }
        
        return global_analysis
    
    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualise les résultats des tests
        
        Args:
            results: Résultats des tests
            save_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Figure matplotlib
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extraction des données
        positions = []
        detection_rates = []
        energy_efficiencies = []
        localization_accuracies = []
        first_detection_steps = []
        
        for pos_str, position_results in results.items():
            if pos_str == 'global_analysis':
                continue
            
            if not position_results:
                continue
            
            # Parse position
            pos = eval(pos_str)  # Convertir string en tuple
            positions.append(pos)
            
            # Moyennes des métriques
            detection_rates.append(np.mean([r['detection_rate'] for r in position_results]))
            energy_efficiencies.append(np.mean([r['energy_efficiency'] for r in position_results]))
            localization_accuracies.append(np.mean([r['localization_accuracy'] for r in position_results]))
            first_detection_steps.append(np.mean([r['first_detection_step'] for r in position_results]))
        
        positions = np.array(positions)
        
        # 1. Carte de chaleur des taux de détection
        ax1 = axes[0, 0]
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], 
                             c=detection_rates, s=200, cmap='RdYlGn', 
                             alpha=0.8, edgecolors='black')
        ax1.set_xlabel('Position X (m)')
        ax1.set_ylabel('Position Y (m)')
        ax1.set_title('Taux de Détection par Position')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Taux de Détection')
        
        # 2. Carte de chaleur de l'efficacité énergétique
        ax2 = axes[0, 1]
        scatter = ax2.scatter(positions[:, 0], positions[:, 1], 
                             c=energy_efficiencies, s=200, cmap='viridis', 
                             alpha=0.8, edgecolors='black')
        ax2.set_xlabel('Position X (m)')
        ax2.set_ylabel('Position Y (m)')
        ax2.set_title('Efficacité Énergétique par Position')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Efficacité Énergétique')
        
        # 3. Carte de chaleur de la précision de localisation
        ax3 = axes[0, 2]
        scatter = ax3.scatter(positions[:, 0], positions[:, 1], 
                             c=localization_accuracies, s=200, cmap='plasma', 
                             alpha=0.8, edgecolors='black')
        ax3.set_xlabel('Position X (m)')
        ax3.set_ylabel('Position Y (m)')
        ax3.set_title('Précision de Localisation par Position')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Précision de Localisation')
        
        # 4. Temps de première détection
        ax4 = axes[1, 0]
        scatter = ax4.scatter(positions[:, 0], positions[:, 1], 
                             c=first_detection_steps, s=200, cmap='coolwarm', 
                             alpha=0.8, edgecolors='black')
        ax4.set_xlabel('Position X (m)')
        ax4.set_ylabel('Position Y (m)')
        ax4.set_title('Temps de Première Détection')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Étape de Première Détection')
        
        # 5. Histogramme des performances
        ax5 = axes[1, 1]
        ax5.hist(detection_rates, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_xlabel('Taux de Détection')
        ax5.set_ylabel('Fréquence')
        ax5.set_title('Distribution des Taux de Détection')
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistiques globales
        ax6 = axes[1, 2]
        global_analysis = results.get('global_analysis', {})
        overall_stats = global_analysis.get('overall_stats', {})
        
        stats_text = f"""
        Tests Totaux: {overall_stats.get('total_tests', 0)}
        
        Taux de Détection:
        Moyenne: {overall_stats.get('mean_detection_rate', 0):.3f}
        Écart-type: {overall_stats.get('std_detection_rate', 0):.3f}
        
        Efficacité Énergétique:
        Moyenne: {overall_stats.get('mean_energy_efficiency', 0):.3f}
        Écart-type: {overall_stats.get('std_energy_efficiency', 0):.3f}
        
        Précision Localisation:
        Moyenne: {overall_stats.get('mean_localization_accuracy', 0):.3f}
        Écart-type: {overall_stats.get('std_localization_accuracy', 0):.3f}
        """
        
        ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax6.set_title('Statistiques Globales')
        ax6.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Sauvegarde les résultats"""
        # Conversion pour JSON
        json_results = {}
        for key, value in results.items():
            if key == 'global_analysis':
                json_results[key] = value
            else:
                # Conversion des résultats de position
                json_results[key] = []
                for result in value:
                    json_result = result.copy()
                    # Conversion des arrays numpy
                    if 'trajectory' in json_result:
                        json_result['trajectory'] = [list(pos) for pos in json_result['trajectory']]
                    if 'detections' in json_result:
                        for det in json_result['detections']:
                            if 'position' in det:
                                det['position'] = det['position'].tolist()
                    json_results[key].append(json_result)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Résultats sauvegardés: {filepath}")


def create_leak_position_test() -> LeakPositionTester:
    """Crée un testeur de positions de fuites avec des positions variées"""
    
    # Positions de fuites à tester (grille 5x5)
    leak_positions = []
    for x in np.linspace(20, 80, 5):
        for y in np.linspace(20, 80, 5):
            leak_positions.append((float(x), float(y)))
    
    # Ajout de positions spéciales
    leak_positions.extend([
        (10, 10),   # Proche du point de départ
        (90, 90),   # Coin opposé
        (50, 10),   # Centre-bas
        (50, 90),   # Centre-haut
        (10, 50),   # Gauche-centre
        (90, 50),   # Droite-centre
    ])
    
    config = LeakTestConfig(
        leak_positions=leak_positions,
        n_runs_per_position=3,  # Réduit pour les tests rapides
        max_steps=300
    )
    
    return LeakPositionTester(config)


if __name__ == "__main__":
    # Test des positions de fuites
    tester = create_leak_position_test()
    
    print("Test de détection sur différentes positions de fuites")
    print("=" * 60)
    
    # Exécution des tests
    results = tester.run_all_tests()
    
    # Visualisation
    fig = tester.plot_results(results)
    plt.show()
    
    # Sauvegarde
    tester.save_results(results, "leak_position_test_results.json")
    
    # Affichage des résultats
    global_analysis = results.get('global_analysis', {})
    overall_stats = global_analysis.get('overall_stats', {})
    
    print("\n" + "=" * 60)
    print("RÉSULTATS GLOBAUX")
    print("=" * 60)
    print(f"Tests totaux: {overall_stats.get('total_tests', 0)}")
    print(f"Taux de détection moyen: {overall_stats.get('mean_detection_rate', 0):.3f} ± {overall_stats.get('std_detection_rate', 0):.3f}")
    print(f"Efficacité énergétique moyenne: {overall_stats.get('mean_energy_efficiency', 0):.3f} ± {overall_stats.get('std_energy_efficiency', 0):.3f}")
    print(f"Précision de localisation moyenne: {overall_stats.get('mean_localization_accuracy', 0):.3f} ± {overall_stats.get('std_localization_accuracy', 0):.3f}")
    
    print("\n✅ Test terminé avec succès!")
