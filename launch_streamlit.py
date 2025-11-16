"""
Lanceur pour l'interface Streamlit HIGHLIGHT+
"""

import subprocess
import sys
import os

def check_dependencies():
    """Vérifie les dépendances nécessaires"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'torch',
        'stable-baselines3',
        'gymnasium'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Packages manquants:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installation des packages manquants...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} installé")
            except subprocess.CalledProcessError:
                print(f"❌ Erreur lors de l'installation de {package}")
                return False
    
    return True

def main():
    """Fonction principale"""
    print("🚁 HIGHLIGHT+ - Lancement de l'interface Streamlit")
    print("=" * 50)
    
    # Vérification des dépendances
    if not check_dependencies():
        print("❌ Impossible de lancer l'application. Vérifiez les dépendances.")
        return
    
    print("✅ Toutes les dépendances sont installées")
    
    # Lancement de Streamlit
    try:
        print("🚀 Lancement de l'interface Streamlit...")
        print("📱 L'interface sera accessible sur: http://localhost:8501")
        print("⏹️ Appuyez sur Ctrl+C pour arrêter")
        print("=" * 50)
        
        # Nettoyage du cache avant lancement
        cache_dir = os.path.join(os.path.expanduser('~'), '.streamlit', 'cache')
        if os.path.exists(cache_dir):
            import shutil
            try:
                shutil.rmtree(cache_dir)
                print("🧹 Cache Streamlit nettoyé")
            except:
                pass
        
        # Lancement de Streamlit avec options de rechargement
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false',
            '--server.runOnSave', 'true',
            '--server.fileWatcherType', 'poll'
        ])
        
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt de l'application")
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")

if __name__ == "__main__":
    main()





