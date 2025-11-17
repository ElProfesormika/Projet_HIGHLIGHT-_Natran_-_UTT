"""
Tests comparatifs pour HIGHLIGHT+ selon le protocole du document LaTeX
Compare HIGHLIGHT+ avec des baselines : recherche aléatoire et balayage systématique
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd

from highlight_plus.simulation.environment import MethaneDetectionEnv, EnvironmentConfig
from highlight_plus.simulation.plume_model import PlumeConfig
from highlight_plus.sensors.tdlas_sensor import TDLASConfig
from highlight_plus.models.teacher_gp import GaussianProcessTeacher, TeacherConfig
from highlight_plus.models.student_rl import StudentRL, StudentConfig


@dataclass
class TestResult:
    """Résultat d'un test de détection"""
    method: str
    detection_rate: float
    localization_precision: float
    energy_consumed: float
    mission_time: float
    score: float


class ComparativeTester:
    """
    Tests comparatifs selon le protocole du document :
    - Baseline A : Recherche aléatoire
    - Baseline B : Balayage systématique
    - Test A : HIGHLIGHT+ (Teacher-Student)
    """
    
    def __init__(self, world_size: Tuple[float, float] = (100.0, 100.0)):
        self.world_size = world_size
        self.results = []
        
    def run_random_search(self, n_runs: int = 10, max_steps: int = 500) -> TestResult:
        """
        Baseline A : Recherche aléatoire
        Le drone se déplace aléatoirement dans l'espace
        """
        print(f"🔀 Exécution de {n_runs} runs de recherche aléatoire...")
        
        detection_rates = []
        localizations = []
        energies = []
        times = []
        
        for run in range(n_runs):
            # Configuration de l'environnement
            env_config = EnvironmentConfig(
                world_size=self.world_size,
                max_steps=max_steps,
                initial_position=(10.0, 10.0),
                initial_altitude=5.0
            )
            
            plume_config = PlumeConfig(
                leak_x=50.0,
                leak_y=50.0,
                leak_intensity=0.3
            )
            
            sensor_config = TDLASConfig()
            
            env = MethaneDetectionEnv(env_config, plume_config, sensor_config)
            obs, info = env.reset()
            
            # Simulation avec actions aléatoires
            for step in range(max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            # Calcul des métriques
            n_detections = len(env.detections)
            detection_rate = n_detections / max_steps
            detection_rates.append(detection_rate)
            
            # Précision de localisation (distance à la vraie source)
            if env.detections:
                detection_pos = env.detections[0]['position'][:2]
                true_pos = np.array([plume_config.leak_x, plume_config.leak_y])
                distance = np.linalg.norm(detection_pos - true_pos)
                localizations.append(distance)
            else:
                localizations.append(float('inf'))
            
            energies.append(env.total_energy_consumed)
            times.append(env.step_count)
        
        # Calcul des moyennes
        avg_detection_rate = np.mean(detection_rates)
        avg_localization = np.mean([d for d in localizations if d != float('inf')]) if any(d != float('inf') for d in localizations) else float('inf')
        avg_energy = np.mean(energies)
        avg_time = np.mean(times)
        
        # Score selon le document : S = (w1*Detection + w2*Precision) / (w3*Energy + w4*Time)
        w1, w2, w3, w4 = 1.0, 1.0, 0.1, 0.01
        score = (w1 * avg_detection_rate + w2 * (1.0 / (1.0 + avg_localization))) / \
                (w3 * avg_energy / 1000.0 + w4 * avg_time)
        
        result = TestResult(
            method="Recherche Aléatoire",
            detection_rate=avg_detection_rate,
            localization_precision=avg_localization,
            energy_consumed=avg_energy,
            mission_time=avg_time,
            score=score
        )
        
        self.results.append(result)
        return result
    
    def run_systematic_scan(self, n_runs: int = 10, max_steps: int = 500) -> TestResult:
        """
        Baseline B : Balayage systématique
        Le drone balaie l'espace selon une grille régulière
        """
        print(f"📐 Exécution de {n_runs} runs de balayage systématique...")
        
        detection_rates = []
        localizations = []
        energies = []
        times = []
        
        for run in range(n_runs):
            env_config = EnvironmentConfig(
                world_size=self.world_size,
                max_steps=max_steps,
                initial_position=(10.0, 10.0),
                initial_altitude=5.0
            )
            
            plume_config = PlumeConfig(
                leak_x=50.0,
                leak_y=50.0,
                leak_intensity=0.3
            )
            
            sensor_config = TDLASConfig()
            
            env = MethaneDetectionEnv(env_config, plume_config, sensor_config)
            obs, info = env.reset()
            
            # Calcul de la grille de balayage
            grid_size = int(np.sqrt(max_steps))
            step_size = self.world_size[0] / grid_size
            
            # Balayage systématique
            for i in range(grid_size):
                for j in range(grid_size):
                    if env.step_count >= max_steps:
                        break
                    
                    # Calcul de la position cible
                    target_x = i * step_size
                    target_y = j * step_size
                    
                    # Action pour se diriger vers la cible
                    current_pos = env.drone_position[:2]
                    direction = np.array([target_x, target_y]) - current_pos
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 0.1:
                        action = np.array([direction[0] / direction_norm,
                                         direction[1] / direction_norm,
                                         0.0], dtype=np.float32)
                        action = np.clip(action, -1, 1)
                    else:
                        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if terminated or truncated:
                        break
            
            # Métriques
            n_detections = len(env.detections)
            detection_rate = n_detections / max_steps
            detection_rates.append(detection_rate)
            
            if env.detections:
                detection_pos = env.detections[0]['position'][:2]
                true_pos = np.array([plume_config.leak_x, plume_config.leak_y])
                distance = np.linalg.norm(detection_pos - true_pos)
                localizations.append(distance)
            else:
                localizations.append(float('inf'))
            
            energies.append(env.total_energy_consumed)
            times.append(env.step_count)
        
        # Calcul des moyennes
        avg_detection_rate = np.mean(detection_rates)
        avg_localization = np.mean([d for d in localizations if d != float('inf')]) if any(d != float('inf') for d in localizations) else float('inf')
        avg_energy = np.mean(energies)
        avg_time = np.mean(times)
        
        # Score
        w1, w2, w3, w4 = 1.0, 1.0, 0.1, 0.01
        score = (w1 * avg_detection_rate + w2 * (1.0 / (1.0 + avg_localization))) / \
                (w3 * avg_energy / 1000.0 + w4 * avg_time)
        
        result = TestResult(
            method="Balayage Systématique",
            detection_rate=avg_detection_rate,
            localization_precision=avg_localization,
            energy_consumed=avg_energy,
            mission_time=avg_time,
            score=score
        )
        
        self.results.append(result)
        return result
    
    def run_highlight_plus(self, n_runs: int = 10, max_steps: int = 500) -> TestResult:
        """
        Test A : HIGHLIGHT+ avec architecture Teacher-Student
        """
        print(f"Execution de {n_runs} runs de HIGHLIGHT+...")
        
        detection_rates = []
        localizations = []
        energies = []
        times = []
        
        for run in range(n_runs):
            env_config = EnvironmentConfig(
                world_size=self.world_size,
                max_steps=max_steps,
                initial_position=(10.0, 10.0),
                initial_altitude=5.0
            )
            
            plume_config = PlumeConfig(
                leak_x=50.0,
                leak_y=50.0,
                leak_intensity=0.3
            )
            
            sensor_config = TDLASConfig()
            
            # Initialisation du Teacher
            teacher_config = TeacherConfig()
            teacher = GaussianProcessTeacher(teacher_config, 
                                             (0, self.world_size[0], 0, self.world_size[1]))
            
            # Initialisation du Student
            student_config = StudentConfig()
            student = StudentRL(state_dim=16, action_dim=3, config=student_config, teacher=teacher)
            
            env = MethaneDetectionEnv(env_config, plume_config, sensor_config)
            obs, info = env.reset()
            
            # Simulation avec HIGHLIGHT+
            for step in range(max_steps):
                # Action du Student
                action = student.select_action(obs, training=True)
                action = np.clip(action, -1, 1)
                
                # Step avec Teacher pour calculer la récompense éco-informative
                obs, reward, terminated, truncated, info = env.step(action, teacher=teacher)
                
                # Mise à jour du Teacher avec la nouvelle observation
                if info.get('concentration') is not None:
                    teacher.add_observation(
                        env.drone_position[0],
                        env.drone_position[1],
                        info['concentration']
                    )
                
                # Stockage de l'expérience et apprentissage
                next_obs = env._get_observation(teacher)
                student.store_experience(obs, action, reward, next_obs, terminated or truncated)
                
                if len(student.replay_buffer) > student.config.learning_starts:
                    student.learn()
                
                obs = next_obs
                student.step_count += 1
                
                if terminated or truncated:
                    break
            
            # Métriques
            n_detections = len(env.detections)
            detection_rate = n_detections / max_steps
            detection_rates.append(detection_rate)
            
            if env.detections:
                detection_pos = env.detections[0]['position'][:2]
                true_pos = np.array([plume_config.leak_x, plume_config.leak_y])
                distance = np.linalg.norm(detection_pos - true_pos)
                localizations.append(distance)
            else:
                localizations.append(float('inf'))
            
            energies.append(env.total_energy_consumed)
            times.append(env.step_count)
        
        # Calcul des moyennes
        avg_detection_rate = np.mean(detection_rates)
        avg_localization = np.mean([d for d in localizations if d != float('inf')]) if any(d != float('inf') for d in localizations) else float('inf')
        avg_energy = np.mean(energies)
        avg_time = np.mean(times)
        
        # Score
        w1, w2, w3, w4 = 1.0, 1.0, 0.1, 0.01
        score = (w1 * avg_detection_rate + w2 * (1.0 / (1.0 + avg_localization))) / \
                (w3 * avg_energy / 1000.0 + w4 * avg_time)
        
        result = TestResult(
            method="HIGHLIGHT+",
            detection_rate=avg_detection_rate,
            localization_precision=avg_localization,
            energy_consumed=avg_energy,
            mission_time=avg_time,
            score=score
        )
        
        self.results.append(result)
        return result
    
    def run_full_comparison(self, n_runs: int = 5, max_steps: int = 200) -> pd.DataFrame:
        """
        Exécute tous les tests comparatifs
        """
        print("=" * 60)
        print("TESTS COMPARATIFS HIGHLIGHT+")
        print("=" * 60)
        
        # Exécution des trois méthodes
        self.run_random_search(n_runs=n_runs, max_steps=max_steps)
        self.run_systematic_scan(n_runs=n_runs, max_steps=max_steps)
        self.run_highlight_plus(n_runs=n_runs, max_steps=max_steps)
        
        # Création du DataFrame de résultats
        df = pd.DataFrame([
            {
                'Méthode': r.method,
                'Taux Détection': r.detection_rate,
                'Précision Localisation (m)': r.localization_precision,
                'Énergie (J)': r.energy_consumed,
                'Temps Mission': r.mission_time,
                'Score Global': r.score
            }
            for r in self.results
        ])
        
        return df
    
    def plot_comparison(self, save_path: str = None):
        """
        Visualise les résultats comparatifs
        """
        if not self.results:
            print("Aucun résultat à visualiser")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        methods = [r.method for r in self.results]
        detection_rates = [r.detection_rate for r in self.results]
        localizations = [r.localization_precision if r.localization_precision != float('inf') else 100 for r in self.results]
        energies = [r.energy_consumed for r in self.results]
        scores = [r.score for r in self.results]
        
        # Graphique 1 : Taux de détection
        axes[0, 0].bar(methods, detection_rates, color=['#ff7f7f', '#7f7fff', '#7fff7f'])
        axes[0, 0].set_title('Taux de Détection')
        axes[0, 0].set_ylabel('Détections / Étape')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Graphique 2 : Précision de localisation
        axes[0, 1].bar(methods, localizations, color=['#ff7f7f', '#7f7fff', '#7fff7f'])
        axes[0, 1].set_title('Précision de Localisation')
        axes[0, 1].set_ylabel('Distance à la source (m)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Graphique 3 : Consommation énergétique
        axes[1, 0].bar(methods, energies, color=['#ff7f7f', '#7f7fff', '#7fff7f'])
        axes[1, 0].set_title('Consommation Énergétique')
        axes[1, 0].set_ylabel('Énergie (J)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Graphique 4 : Score global
        axes[1, 1].bar(methods, scores, color=['#ff7f7f', '#7f7fff', '#7fff7f'])
        axes[1, 1].set_title('Score Global (selon document)')
        axes[1, 1].set_ylabel('Score S')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()


if __name__ == "__main__":
    # Exécution des tests comparatifs
    tester = ComparativeTester(world_size=(100.0, 100.0))
    
    # Tests complets (réduire n_runs pour des tests rapides)
    df_results = tester.run_full_comparison(n_runs=3, max_steps=200)
    
    print("\n" + "=" * 60)
    print("RESULTATS COMPARATIFS")
    print("=" * 60)
    print(df_results.to_string(index=False))
    
    # Visualisation
    tester.plot_comparison(save_path="comparison_results.png")





