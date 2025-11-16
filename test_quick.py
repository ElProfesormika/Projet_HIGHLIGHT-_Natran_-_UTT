"""
Test rapide de HIGHLIGHT+ - Vérification des composants principaux
"""

import numpy as np
import matplotlib.pyplot as plt

def test_imports():
    """Test des imports"""
    print("🔍 Test des imports...")
    
    try:
        from highlight_plus.simulation.plume_model import MethanePlume, PlumeConfig
        print("   ✅ Modèle de panache importé")
    except Exception as e:
        print(f"   ❌ Erreur import panache: {e}")
        return False
    
    try:
        from highlight_plus.sensors.tdlas_sensor import TDLASSensor, TDLASConfig
        print("   ✅ Capteur TDLAS importé")
    except Exception as e:
        print(f"   ❌ Erreur import capteur: {e}")
        return False
    
    try:
        from highlight_plus.models.teacher_gp import GaussianProcessTeacher, TeacherConfig
        print("   ✅ Expert (Teacher) importé")
    except Exception as e:
        print(f"   ❌ Erreur import teacher: {e}")
        return False
    
    try:
        from highlight_plus.models.student_rl import StudentRL, StudentConfig
        print("   ✅ Apprenti (Student) importé")
    except Exception as e:
        print(f"   ❌ Erreur import student: {e}")
        return False
    
    try:
        from highlight_plus.simulation.environment import MethaneDetectionEnv, EnvironmentConfig
        print("   ✅ Environnement de simulation importé")
    except Exception as e:
        print(f"   ❌ Erreur import environnement: {e}")
        return False
    
    return True


def test_plume():
    """Test du modèle de panache"""
    print("\n🌪️  Test du modèle de panache...")
    
    try:
        from highlight_plus.simulation.plume_model import create_test_plume
        
        plume = create_test_plume()
        
        # Test de concentration
        conc = plume.concentration(50, 50)
        print(f"   ✅ Concentration au centre: {conc:.4f} kg/m³")
        
        # Test du gradient
        grad_x, grad_y = plume.gradient(50, 50)
        print(f"   ✅ Gradient: ({grad_x:.4f}, {grad_y:.4f})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur test panache: {e}")
        return False


def test_sensor():
    """Test du capteur TDLAS"""
    print("\n📡 Test du capteur TDLAS...")
    
    try:
        from highlight_plus.sensors.tdlas_sensor import create_test_sensor
        
        sensor = create_test_sensor()
        
        # Test de mesure
        measured, detected = sensor.measure_concentration(0.1, 10.0)
        print(f"   ✅ Mesure: {measured:.4f}, Détecté: {detected}")
        
        # Test SNR
        snr = sensor.get_signal_to_noise_ratio(0.1, 10.0)
        print(f"   ✅ SNR: {snr:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur test capteur: {e}")
        return False


def test_teacher():
    """Test de l'Expert (Teacher)"""
    print("\n🧠 Test de l'Expert (Teacher)...")
    
    try:
        from highlight_plus.models.teacher_gp import create_test_teacher
        
        teacher = create_test_teacher()
        
        # Test d'ajout d'observation
        teacher.add_observation(10, 10, 0.1)
        teacher.add_observation(20, 20, 0.2)
        print("   ✅ Observations ajoutées")
        
        # Test de sélection de point
        next_x, next_y = teacher.select_next_point(15, 15)
        print(f"   ✅ Prochain point: ({next_x:.1f}, {next_y:.1f})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur test teacher: {e}")
        return False


def test_student():
    """Test de l'Apprenti (Student)"""
    print("\n🎓 Test de l'Apprenti (Student)...")
    
    try:
        from highlight_plus.models.student_rl import create_test_student
        
        student = create_test_student()
        
        # Test de sélection d'action
        state = np.random.randn(11)
        action = student.select_action(state)
        print(f"   ✅ Action sélectionnée: {action}")
        
        # Test de stockage d'expérience
        student.store_experience(state, action, 1.0, state, False)
        print("   ✅ Expérience stockée")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur test student: {e}")
        return False


def test_environment():
    """Test de l'environnement"""
    print("\n🌍 Test de l'environnement...")
    
    try:
        from highlight_plus.simulation.environment import create_test_environment
        
        env = create_test_environment()
        
        # Test de reset
        obs, info = env.reset()
        print(f"   ✅ Reset: obs shape {obs.shape}, info keys {list(info.keys())}")
        
        # Test de step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   ✅ Step: reward {reward:.3f}, terminated {terminated}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur test environnement: {e}")
        return False


def main():
    """Fonction principale de test"""
    print("🚀 Test rapide de HIGHLIGHT+")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_plume,
        test_sensor,
        test_teacher,
        test_student,
        test_environment
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 RÉSULTATS: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont passés! HIGHLIGHT+ est prêt.")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les dépendances.")
    
    return passed == total


if __name__ == "__main__":
    main()









