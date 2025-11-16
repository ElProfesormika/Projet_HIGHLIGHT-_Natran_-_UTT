"""
Lanceur de l'interface graphique HIGHLIGHT+
Interface utilisateur pour paramétrer et lancer les simulations
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import gymnasium
    except ImportError:
        missing_deps.append("gymnasium")
    
    if missing_deps:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Dépendances manquantes",
            f"Les dépendances suivantes sont manquantes:\n\n" +
            "\n".join(f"• {dep}" for dep in missing_deps) +
            "\n\nVeuillez les installer avec:\n" +
            f"pip install {' '.join(missing_deps)}"
        )
        return False
    
    return True

def main():
    """Fonction principale"""
    print("🚁 HIGHLIGHT+ - Lancement de l'interface graphique")
    print("=" * 50)
    
    # Vérification des dépendances
    if not check_dependencies():
        print("❌ Dépendances manquantes. Veuillez les installer.")
        return
    
    print("✅ Toutes les dépendances sont installées")
    
    try:
        # Import de l'interface graphique
        from highlight_plus.gui.main_window import HighlightPlusGUI
        
        # Création de la fenêtre principale
        root = tk.Tk()
        app = HighlightPlusGUI(root)
        
        print("✅ Interface graphique initialisée")
        print("🎉 HIGHLIGHT+ est prêt!")
        
        # Lancement de l'interface
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")
        
        # Affichage de l'erreur dans une boîte de dialogue
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Erreur de lancement",
            f"Erreur lors du lancement de l'interface:\n\n{e}\n\n" +
            "Vérifiez que tous les fichiers sont présents et que les dépendances sont installées."
        )

if __name__ == "__main__":
    main()
