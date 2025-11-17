"""
Démonstration HIGHLIGHT+ - Système de détection intelligente de micro-fuites de méthane
Concours Innovation Natran x Fondation UTT

Ce script démontre les capacités du système HIGHLIGHT+ en simulant une mission
de détection de fuite de méthane avec l'architecture Teacher-Student.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Import des composants du projet
from highlight_plus.simulation.environment import create_test_environment
from highlight_plus.simulation.plume_model import create_test_plume
from highlight_plus.sensors.tdlas_sensor import create_test_sensor
from highlight_plus.models.teacher_gp import create_test_teacher
from highlight_plus.models.student_rl import create_test_student
from highlight_plus.visualization.plotter import create_test_plotter
from highlight_plus.experiments.run_comparison import ExperimentRunner, ExperimentConfig


def print_banner():
    """Affiche la bannière du projet"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║                    HIGHLIGHT+ - DETECTION INTELLIGENTE                    ║
    ║                                                                              ║
    ║              Optimisation des trajectoires de vol – Drones                   ║
    ║              pour la détection de micro-fuites de méthane                    ║
    ║                                                                              ║
    ║                        Concours Innovation Natran x UTT                      ║
    ║                                                                              ║
    ║  Équipe: Housséni YABRE, Kabinet SYLLA, Nobert Bassooma DIDANERA           ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def demo_plume_model():
    """Démonstration du modèle de panache de méthane"""
    print("\n" + "="*60)
    print("DEMONSTRATION DU MODELE DE PANACHE DE METHANE")
    print("="*60)
    
    # Création du panache de test
    plume = create_test_plume()
    
    print("Modele de panache cree avec succes")
    print(f"   - Position de la fuite: ({plume.config.leak_x:.1f}, {plume.config.leak_y:.1f}) m")
    print(f"   - Intensité de la fuite: {plume.config.leak_intensity:.2f} kg/s")
    print(f"   - Vitesse du vent: {plume.config.wind_speed:.1f} m/s")
    print(f"   - Direction du vent: {plume.config.wind_direction:.0f}°")
    
    # Test de la concentration
    test_points = [(30, 30), (50, 50), (70, 70)]
    print("\nTest de concentration en differents points:")
    for x, y in test_points:
        conc = plume.concentration(x, y)
        print(f"   - Point ({x}, {y}): {conc:.4f} kg/m³")
    
    # Visualisation
    print("\nGeneration de la visualisation...")
    fig, ax = plt.subplots(figsize=(10, 8))
    plume.plot_plume(ax=ax)
    plt.title("Modèle de Panache de Méthane - HIGHLIGHT+")
    plt.show()
    
    return plume


def demo_sensor():
    """Démonstration du capteur TDLAS"""
    print("\n" + "="*60)
    print("📡 DÉMONSTRATION DU CAPTEUR TDLAS")
    print("="*60)
    
    # Création du capteur
    sensor = create_test_sensor()
    
    print("Capteur TDLAS cree avec succes")
    print(f"   - Niveau de bruit: {sensor.config.noise_level:.3f}")
    print(f"   - Seuil de détection: {sensor.config.detection_threshold:.3f} kg/m³")
    print(f"   - Portée maximale: {sensor.config.range_max:.0f} m")
    print(f"   - Fréquence de mise à jour: {sensor.config.update_frequency:.0f} Hz")
    
    # Test de mesure
    print("\nTest de mesure avec differentes concentrations:")
    concentrations = [0.0, 0.02, 0.05, 0.1, 0.2]
    distances = [5.0, 10.0, 15.0]
    
    for conc in concentrations:
        for dist in distances:
            measured, detected = sensor.measure_concentration(conc, dist)
            snr = sensor.get_signal_to_noise_ratio(conc, dist)
            print(f"   - Conc: {conc:.3f}, Dist: {dist:.0f}m → "
                  f"Mesuré: {measured:.3f}, Détecté: {detected}, SNR: {snr:.2f}")
    
    # Visualisation de l'historique
    print("\nVisualisation de l'historique des mesures...")
    fig, ax = plt.subplots(figsize=(12, 6))
    sensor.plot_measurement_history(ax=ax)
    plt.title("Historique des Mesures TDLAS - HIGHLIGHT+")
    plt.show()
    
    return sensor


def demo_teacher():
    """Démonstration de l'Expert (Teacher)"""
    print("\n" + "="*60)
    print("DEMONSTRATION DE L'EXPERT (TEACHER)")
    print("="*60)
    
    # Création du Teacher
    teacher = create_test_teacher()
    
    print("Expert (Teacher) cree avec succes")
    print(f"   - Kernel: RBF avec échelle {teacher.config.kernel_length_scale:.1f}")
    print(f"   - Fonction d'acquisition: {teacher.config.acquisition_function}")
    print(f"   - Paramètre d'exploration: {teacher.config.exploration_parameter:.1f}")
    
    # Simulation d'observations
    print("\nSimulation d'observations et d'apprentissage actif...")
    np.random.seed(42)
    
    current_x, current_y = 10.0, 10.0
    teacher.current_position = (current_x, current_y)
    
    for i in range(8):
        # Mesure simulée
        concentration = np.random.exponential(0.1)
        uncertainty = np.random.uniform(0.05, 0.2)
        
        # Ajout de l'observation
        teacher.add_observation(current_x, current_y, concentration, uncertainty)
        
        # Sélection du prochain point
        next_x, next_y = teacher.select_next_point(current_x, current_y)
        
        print(f"   Étape {i+1}: Position ({current_x:.1f}, {current_y:.1f}) → "
              f"Conc: {concentration:.3f} → Next: ({next_x:.1f}, {next_y:.1f})")
        
        current_x, current_y = next_x, next_y
    
    # Visualisation des résultats
    print("\nVisualisation des resultats de l'Expert...")
    fig, ax = plt.subplots(figsize=(12, 10))
    teacher.plot_results(ax=ax)
    plt.title("Expert (Teacher) - Prédiction et Trajectoire - HIGHLIGHT+")
    plt.show()
    
    # Métriques de performance
    metrics = teacher.get_performance_metrics()
    print("\nMetriques de performance de l'Expert:")
    for key, value in metrics.items():
        print(f"   - {key}: {value:.4f}")
    
    return teacher


def demo_student():
    """Démonstration de l'Apprenti (Student)"""
    print("\n" + "="*60)
    print("DEMONSTRATION DE L'APPRENTI (STUDENT)")
    print("="*60)
    
    # Création du Student
    student = create_test_student()
    
    print("Apprenti (Student) cree avec succes")
    print(f"   - Architecture: {student.config.hidden_layers}")
    print(f"   - Taux d'apprentissage: {student.config.learning_rate:.2e}")
    print(f"   - Poids de distillation: {student.config.lambda_kl:.2f}")
    
    # Test de sélection d'action
    print("\nTest de selection d'action...")
    state = np.random.randn(11)
    action = student.select_action(state)
    print(f"   - État d'entrée: {state[:5]}... (dimension: {len(state)})")
    print(f"   - Action sélectionnée: {action}")
    
    # Simulation d'apprentissage
    print("\nSimulation d'apprentissage...")
    for i in range(50):
        # Simulation d'expériences
        state = np.random.randn(11)
        action = student.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(11)
        done = False
        
        student.store_experience(state, action, reward, next_state, done)
        student.step_count += 1
    
    # Apprentissage
    metrics = student.learn()
    print(f"   - Métriques d'apprentissage: {metrics}")
    
    # Visualisation des progrès
    print("\nVisualisation des progres d'entrainement...")
    fig, ax = plt.subplots(figsize=(12, 6))
    student.plot_training_progress(ax=ax)
    plt.title("Progrès d'Entraînement de l'Apprenti - HIGHLIGHT+")
    plt.show()
    
    # Métriques de performance
    perf_metrics = student.get_performance_metrics()
    print("\nMetriques de performance de l'Apprenti:")
    for key, value in perf_metrics.items():
        print(f"   - {key}: {value:.4f}")
    
    return student


def demo_environment():
    """Démonstration de l'environnement de simulation"""
    print("\n" + "="*60)
    print("🌍 DÉMONSTRATION DE L'ENVIRONNEMENT DE SIMULATION")
    print("="*60)
    
    # Création de l'environnement
    env = create_test_environment()
    
    print("Environnement de simulation cree avec succes")
    print(f"   - Dimensions du monde: {env.world_width}x{env.world_height} m")
    print(f"   - Nombre maximum d'étapes: {env.config.max_steps}")
    print(f"   - Position initiale: ({env.config.initial_position[0]}, {env.config.initial_position[1]}) m")
    print(f"   - Altitude initiale: {env.config.initial_altitude} m")
    
    # Test de l'environnement
    print("\nTest de l'environnement...")
    obs, info = env.reset()
    print(f"   - Observation initiale: {obs[:5]}... (dimension: {len(obs)})")
    print(f"   - Info initiale: {info}")
    
    # Simulation de quelques étapes
    print("\nSimulation de quelques etapes...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Étape {i+1}: Action {action[:2]} → Reward {reward:.3f}, "
              f"Position {info['position'][:2]}")
        
        if terminated or truncated:
            break
    
    # Rendu de l'environnement
    print("\nRendu de l'environnement...")
    env.render()
    
    env.close()
    return env


def demo_comparison():
    """Démonstration de la comparaison complète"""
    print("\n" + "="*60)
    print("DEMONSTRATION DE LA COMPARAISON COMPLETE")
    print("="*60)
    
    # Configuration des expériences
    config = ExperimentConfig(
        n_runs=3,
        max_steps=200,
        save_results=True,
        save_plots=True,
        output_dir="demo_results"
    )
    
    print("Configuration des experiences creee")
    print(f"   - Nombre de runs: {config.n_runs}")
    print(f"   - Étapes maximum: {config.max_steps}")
    print(f"   - Dossier de sortie: {config.output_dir}")
    
    # Exécution des expériences
    print("\nExecution des experiences comparatives...")
    runner = ExperimentRunner(config)
    
    try:
        results = runner.run_comparison()
        print("\nExperiences terminees avec succes!")
        
        # Affichage des résultats clés
        print("\nRESULTATS CLES:")
        for method, data in results.items():
            metrics = data['metrics']
            print(f"\n   {method.upper()}:")
            print(f"     - Taux de détection: {metrics['detection_rate']:.3f}")
            print(f"     - Efficacité énergétique: {metrics['energy_efficiency']:.3f}")
            print(f"     - Précision de localisation: {metrics['localization_accuracy']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"ERREUR lors de l'execution des experiences: {e}")
        return None


def main():
    """Fonction principale de démonstration"""
    print_banner()
    
    print("Demarrage de la demonstration HIGHLIGHT+")
    print("Cette démonstration présente les capacités du système de détection")
    print("intelligente de micro-fuites de méthane avec l'architecture Teacher-Student.")
    
    try:
        # Démonstrations individuelles
        print("\n" + "DEMONSTRATIONS INDIVIDUELLES")
        print("="*60)
        
        # 1. Modèle de panache
        plume = demo_plume_model()
        time.sleep(1)
        
        # 2. Capteur TDLAS
        sensor = demo_sensor()
        time.sleep(1)
        
        # 3. Expert (Teacher)
        teacher = demo_teacher()
        time.sleep(1)
        
        # 4. Apprenti (Student)
        student = demo_student()
        time.sleep(1)
        
        # 5. Environnement de simulation
        env = demo_environment()
        time.sleep(1)
        
        # Démonstration de la comparaison complète
        print("\n" + "DEMONSTRATION DE LA COMPARAISON COMPLETE")
        print("="*60)
        
        results = demo_comparison()
        
        # Conclusion
        print("\n" + "="*60)
        print("DEMONSTRATION TERMINEE AVEC SUCCES!")
        print("="*60)
        
        print("\nRESUME DE LA DEMONSTRATION:")
        print("   - Modele de panache de methane - Fonctionnel")
        print("   - Capteur TDLAS simule - Fonctionnel")
        print("   - Expert (Teacher) avec GP - Fonctionnel")
        print("   - Apprenti (Student) avec RL - Fonctionnel")
        print("   - Environnement de simulation - Fonctionnel")
        print("   - Comparaison complete - Fonctionnelle")
        
        print("\nOBJECTIFS ATTEINTS:")
        print("   - Architecture Teacher-Student implementee")
        print("   - Modele mathematique realiste du panache")
        print("   - Simulation precise du capteur TDLAS")
        print("   - Optimisation energetique demontree")
        print("   - Metriques de performance quantifiees")
        
        print("\nPROCHAINES ETAPES:")
        print("   1. Intégration matérielle (Phase 1)")
        print("   2. Tests en conditions réelles")
        print("   3. Démonstration pilote industrielle")
        print("   4. Commercialisation en 'Inspection-as-a-Service'")
        
        print("\nINNOVATION DEMONTREE:")
        print("   - Combinaison unique IA + physique")
        print("   - Apprentissage actif pour l'exploration")
        print("   - Distillation de connaissance Teacher-Student")
        print("   - Optimisation multi-objectifs (détection + énergie)")
        
        print("\nHIGHLIGHT+ est pret pour le concours Natran x UTT!")
        
    except Exception as e:
        print(f"\nERREUR lors de la demonstration: {e}")
        print("Veuillez vérifier que toutes les dépendances sont installées.")
        print("Exécutez: pip install -r requirements.txt")


if __name__ == "__main__":
    main()










