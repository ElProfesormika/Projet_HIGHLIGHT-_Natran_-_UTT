"""
Script de lancement pour HIGHLIGHT+ Streamlit
Garantit l'utilisation du bon environnement Python
"""

import sys
import subprocess
import os

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    print("🔍 Vérification des dépendances...")
    
    required_modules = {
        'gymnasium': 'gymnasium',
        'streamlit': 'streamlit',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'plotly': 'plotly',
        'sklearn': 'scikit-learn',
        'torch': 'torch'
    }
    
    missing = []
    for module_name, package_name in required_modules.items():
        try:
            __import__(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - MANQUANT")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️ Modules manquants: {', '.join(missing)}")
        print("Installation...")
        for package in missing:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"   ✅ {package} installé")
            except:
                print(f"   ❌ Erreur installation {package}")
        return False
    
    print("✅ Toutes les dépendances sont installées\n")
    return True

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🚁 HIGHLIGHT+ - Lancement de l'Application Streamlit")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}\n")
    
    # Vérification des dépendances
    if not check_dependencies():
        print("⚠️ Certaines dépendances manquent. L'application peut ne pas fonctionner correctement.")
        response = input("Continuer quand même ? (o/n): ")
        if response.lower() != 'o':
            return
    
    # Vérification que streamlit_app.py existe
    if not os.path.exists('streamlit_app.py'):
        print("❌ Erreur: streamlit_app.py introuvable")
        print("Assurez-vous d'être dans le répertoire du projet")
        return
    
    # Lancement de Streamlit
    print("🚀 Lancement de l'interface Streamlit...")
    print("📱 L'interface sera accessible sur: http://localhost:8501")
    print("⏹️ Appuyez sur Ctrl+C pour arrêter")
    print("=" * 60)
    print()
    
    try:
        # Utiliser python -m streamlit pour garantir le bon environnement
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt de l'application")
    except Exception as e:
        print(f"\n❌ Erreur lors du lancement: {e}")
        print("\n💡 Essayez manuellement:")
        print(f"   {sys.executable} -m streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()


