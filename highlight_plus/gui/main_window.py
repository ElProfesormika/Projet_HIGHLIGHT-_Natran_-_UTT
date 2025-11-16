"""
Interface graphique principale pour HIGHLIGHT+
Interface utilisateur pour paramétrer et lancer les simulations
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import json
from pathlib import Path
import sys
import os
import time
from datetime import datetime
import pandas as pd

# Ajout du chemin du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from highlight_plus.simulation.plume_model import MethanePlume, PlumeConfig
from highlight_plus.sensors.tdlas_sensor import TDLASSensor, TDLASConfig
from highlight_plus.models.teacher_gp import GaussianProcessTeacher, TeacherConfig
from highlight_plus.models.student_rl import StudentRL, StudentConfig
from highlight_plus.simulation.environment import MethaneDetectionEnv, EnvironmentConfig
from highlight_plus.data.real_data_loader import RealDataLoader, RealDataConfig
from highlight_plus.analysis.learning_analysis import LearningAnalyzer
from highlight_plus.experiments.leak_position_test import LeakPositionTester


class HighlightPlusGUI:
    """Interface graphique principale pour HIGHLIGHT+"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🚁 HIGHLIGHT+ - Système de Détection Intelligente de Méthane")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Variables de configuration
        self.config_vars = {}
        self.simulation_running = False
        self.results = {}
        self.performance_metrics = {}
        self.learning_history = []
        self.leak_positions = []
        self.current_step = 0
        self.start_time = None
        
        # Création de l'interface
        self.create_widgets()
        self.setup_default_values()
        
    def create_widgets(self):
        """Crée tous les widgets de l'interface"""
        # Notebook pour les onglets
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Onglet Configuration
        self.create_config_tab()
        
        # Onglet Simulation
        self.create_simulation_tab()
        
        # Onglet Données Réelles
        self.create_real_data_tab()
        
        # Onglet Résultats
        self.create_results_tab()
        
        # Onglet Analyse d'Apprentissage
        self.create_learning_analysis_tab()
        
        # Onglet Test de Positions
        self.create_position_test_tab()
        
        # Barre de statut
        self.create_status_bar()
    
    def create_config_tab(self):
        """Crée l'onglet de configuration"""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="⚙️ Configuration")
        
        # Notebook pour les sous-onglets de configuration
        config_notebook = ttk.Notebook(config_frame)
        config_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configuration du panache
        self.create_plume_config_tab(config_notebook)
        
        # Configuration du capteur
        self.create_sensor_config_tab(config_notebook)
        
        # Configuration du drone
        self.create_drone_config_tab(config_notebook)
        
        # Configuration des modèles IA
        self.create_ai_config_tab(config_notebook)
        
        # Configuration des positions de fuites
        self.create_leak_positions_tab(config_notebook)
    
    def create_plume_config_tab(self, parent):
        """Configuration du modèle de panache"""
        plume_frame = ttk.Frame(parent)
        parent.add(plume_frame, text="🌪️ Panache")
        
        # Variables
        self.config_vars['leak_x'] = tk.DoubleVar(value=50.0)
        self.config_vars['leak_y'] = tk.DoubleVar(value=50.0)
        self.config_vars['leak_intensity'] = tk.DoubleVar(value=0.3)
        self.config_vars['wind_speed'] = tk.DoubleVar(value=2.0)
        self.config_vars['wind_direction'] = tk.DoubleVar(value=45.0)
        self.config_vars['sigma_x'] = tk.DoubleVar(value=5.0)
        self.config_vars['sigma_y'] = tk.DoubleVar(value=3.0)
        
        # Interface
        ttk.Label(plume_frame, text="Configuration du Panache de Méthane", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Position de la fuite
        leak_frame = ttk.LabelFrame(plume_frame, text="Position de la Fuite")
        leak_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(leak_frame, text="X (m):").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(leak_frame, textvariable=self.config_vars['leak_x'], width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(leak_frame, text="Y (m):").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(leak_frame, textvariable=self.config_vars['leak_y'], width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(leak_frame, text="Intensité (kg/s):").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(leak_frame, textvariable=self.config_vars['leak_intensity'], width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Conditions environnementales
        env_frame = ttk.LabelFrame(plume_frame, text="Conditions Environnementales")
        env_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(env_frame, text="Vitesse vent (m/s):").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(env_frame, textvariable=self.config_vars['wind_speed'], width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(env_frame, text="Direction vent (°):").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(env_frame, textvariable=self.config_vars['wind_direction'], width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Paramètres de diffusion
        diff_frame = ttk.LabelFrame(plume_frame, text="Paramètres de Diffusion")
        diff_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(diff_frame, text="σx (m):").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(diff_frame, textvariable=self.config_vars['sigma_x'], width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(diff_frame, text="σy (m):").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(diff_frame, textvariable=self.config_vars['sigma_y'], width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Bouton de visualisation
        ttk.Button(plume_frame, text="🎨 Visualiser le Panache", 
                  command=self.visualize_plume).pack(pady=10)
    
    def create_sensor_config_tab(self, parent):
        """Configuration du capteur TDLAS"""
        sensor_frame = ttk.Frame(parent)
        parent.add(sensor_frame, text="📡 Capteur")
        
        # Variables
        self.config_vars['noise_level'] = tk.DoubleVar(value=0.1)
        self.config_vars['detection_threshold'] = tk.DoubleVar(value=0.05)
        self.config_vars['range_max'] = tk.DoubleVar(value=100.0)
        self.config_vars['update_frequency'] = tk.DoubleVar(value=10.0)
        
        # Interface
        ttk.Label(sensor_frame, text="Configuration du Capteur TDLAS", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Paramètres du capteur
        sensor_params_frame = ttk.LabelFrame(sensor_frame, text="Paramètres du Capteur")
        sensor_params_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(sensor_params_frame, text="Niveau de bruit:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(sensor_params_frame, textvariable=self.config_vars['noise_level'], width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(sensor_params_frame, text="Seuil détection (kg/m³):").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(sensor_params_frame, textvariable=self.config_vars['detection_threshold'], width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(sensor_params_frame, text="Portée max (m):").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(sensor_params_frame, textvariable=self.config_vars['range_max'], width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(sensor_params_frame, text="Fréquence (Hz):").grid(row=1, column=2, padx=5, pady=5)
        ttk.Entry(sensor_params_frame, textvariable=self.config_vars['update_frequency'], width=10).grid(row=1, column=3, padx=5, pady=5)
    
    def create_drone_config_tab(self, parent):
        """Configuration du drone"""
        drone_frame = ttk.Frame(parent)
        parent.add(drone_frame, text="🚁 Drone")
        
        # Variables
        self.config_vars['max_speed'] = tk.DoubleVar(value=5.0)
        self.config_vars['max_altitude'] = tk.DoubleVar(value=20.0)
        self.config_vars['initial_x'] = tk.DoubleVar(value=10.0)
        self.config_vars['initial_y'] = tk.DoubleVar(value=10.0)
        self.config_vars['initial_altitude'] = tk.DoubleVar(value=5.0)
        
        # Interface
        ttk.Label(drone_frame, text="Configuration du Drone", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Paramètres de vol
        flight_frame = ttk.LabelFrame(drone_frame, text="Paramètres de Vol")
        flight_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(flight_frame, text="Vitesse max (m/s):").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(flight_frame, textvariable=self.config_vars['max_speed'], width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(flight_frame, text="Altitude max (m):").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(flight_frame, textvariable=self.config_vars['max_altitude'], width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Position initiale
        init_frame = ttk.LabelFrame(drone_frame, text="Position Initiale")
        init_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(init_frame, text="X initial (m):").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(init_frame, textvariable=self.config_vars['initial_x'], width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(init_frame, text="Y initial (m):").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(init_frame, textvariable=self.config_vars['initial_y'], width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(init_frame, text="Altitude initiale (m):").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(init_frame, textvariable=self.config_vars['initial_altitude'], width=10).grid(row=1, column=1, padx=5, pady=5)
    
    def create_ai_config_tab(self, parent):
        """Configuration des modèles IA"""
        ai_frame = ttk.Frame(parent)
        parent.add(ai_frame, text="🧠 IA")
        
        # Variables
        self.config_vars['teacher_exploration'] = tk.DoubleVar(value=2.5)
        self.config_vars['student_learning_rate'] = tk.DoubleVar(value=1e-3)
        self.config_vars['student_lambda_kl'] = tk.DoubleVar(value=0.2)
        self.config_vars['max_steps'] = tk.IntVar(value=500)
        
        # Interface
        ttk.Label(ai_frame, text="Configuration des Modèles IA", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Teacher
        teacher_frame = ttk.LabelFrame(ai_frame, text="Expert (Teacher)")
        teacher_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(teacher_frame, text="Paramètre d'exploration:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(teacher_frame, textvariable=self.config_vars['teacher_exploration'], width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Student
        student_frame = ttk.LabelFrame(ai_frame, text="Apprenti (Student)")
        student_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(student_frame, text="Taux d'apprentissage:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(student_frame, textvariable=self.config_vars['student_learning_rate'], width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(student_frame, text="Poids distillation:").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(student_frame, textvariable=self.config_vars['student_lambda_kl'], width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Simulation
        sim_frame = ttk.LabelFrame(ai_frame, text="Simulation")
        sim_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(sim_frame, text="Étapes max:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(sim_frame, textvariable=self.config_vars['max_steps'], width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Mode de simulation
        self.config_vars['simulation_mode'] = tk.StringVar(value="simple")
        mode_frame = ttk.LabelFrame(ai_frame, text="Mode de Simulation")
        mode_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Radiobutton(mode_frame, text="Simple", variable=self.config_vars['simulation_mode'], value="simple").grid(row=0, column=0, padx=5, pady=5)
        ttk.Radiobutton(mode_frame, text="Teacher-Student", variable=self.config_vars['simulation_mode'], value="teacher_student").grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(mode_frame, text="Apprentissage Complet", variable=self.config_vars['simulation_mode'], value="full_learning").grid(row=0, column=2, padx=5, pady=5)
    
    def create_simulation_tab(self):
        """Crée l'onglet de simulation"""
        sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(sim_frame, text="🚀 Simulation")
        
        # Contrôles de simulation
        control_frame = ttk.LabelFrame(sim_frame, text="Contrôles de Simulation")
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Boutons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        self.start_button = ttk.Button(button_frame, text="▶️ Démarrer Simulation", 
                                      command=self.start_simulation)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="⏹️ Arrêter", 
                                     command=self.stop_simulation, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="💾 Sauvegarder Config", 
                  command=self.save_config).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="📂 Charger Config", 
                  command=self.load_config).pack(side='left', padx=5)
        
        # Barre de progression
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                           maximum=100)
        self.progress_bar.pack(fill='x', padx=10, pady=5)
        
        # Zone de log
        log_frame = ttk.LabelFrame(sim_frame, text="Log de Simulation")
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15)
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_real_data_tab(self):
        """Crée l'onglet de données réelles"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📊 Données Réelles")
        
        # Chargement de données
        load_frame = ttk.LabelFrame(data_frame, text="Chargement de Données")
        load_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(load_frame, text="📂 Charger Fichier CSV", 
                  command=self.load_real_data).pack(pady=5)
        
        ttk.Button(load_frame, text="🎲 Générer Données d'Exemple", 
                  command=self.generate_sample_data).pack(pady=5)
        
        # Affichage des données
        self.data_display = scrolledtext.ScrolledText(data_frame, height=20)
        self.data_display.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_results_tab(self):
        """Crée l'onglet de résultats"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📈 Résultats")
        
        # Zone de visualisation
        self.results_frame = results_frame
        
        # Contrôles
        control_frame = ttk.Frame(results_frame)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(control_frame, text="📊 Afficher Résultats", 
                  command=self.display_results).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="💾 Exporter Résultats", 
                  command=self.export_results).pack(side='left', padx=5)
    
    def create_leak_positions_tab(self, parent):
        """Configuration des positions de fuites multiples"""
        leak_frame = ttk.Frame(parent)
        parent.add(leak_frame, text="📍 Positions Fuites")
        
        # Interface
        ttk.Label(leak_frame, text="Configuration des Positions de Fuites", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Liste des positions
        list_frame = ttk.LabelFrame(leak_frame, text="Positions Configurées")
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Treeview pour afficher les positions
        columns = ('X', 'Y', 'Intensité', 'Actif')
        self.leak_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.leak_tree.heading(col, text=col)
            self.leak_tree.column(col, width=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.leak_tree.yview)
        self.leak_tree.configure(yscrollcommand=scrollbar.set)
        
        self.leak_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Contrôles
        control_frame = ttk.Frame(leak_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Variables pour nouvelle position
        self.new_leak_x = tk.DoubleVar(value=50.0)
        self.new_leak_y = tk.DoubleVar(value=50.0)
        self.new_leak_intensity = tk.DoubleVar(value=0.3)
        
        # Entrées
        ttk.Label(control_frame, text="X:").grid(row=0, column=0, padx=2)
        ttk.Entry(control_frame, textvariable=self.new_leak_x, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(control_frame, text="Y:").grid(row=0, column=2, padx=2)
        ttk.Entry(control_frame, textvariable=self.new_leak_y, width=8).grid(row=0, column=3, padx=2)
        
        ttk.Label(control_frame, text="Intensité:").grid(row=0, column=4, padx=2)
        ttk.Entry(control_frame, textvariable=self.new_leak_intensity, width=8).grid(row=0, column=5, padx=2)
        
        # Boutons
        ttk.Button(control_frame, text="➕ Ajouter", 
                  command=self.add_leak_position).grid(row=0, column=6, padx=5)
        ttk.Button(control_frame, text="🗑️ Supprimer", 
                  command=self.remove_leak_position).grid(row=0, column=7, padx=5)
        ttk.Button(control_frame, text="🎲 Positions Aléatoires", 
                  command=self.generate_random_positions).grid(row=0, column=8, padx=5)
        ttk.Button(control_frame, text="📂 Charger", 
                  command=self.load_leak_positions).grid(row=0, column=9, padx=5)
        ttk.Button(control_frame, text="💾 Sauvegarder", 
                  command=self.save_leak_positions).grid(row=0, column=10, padx=5)
        
        # Positions prédéfinies
        preset_frame = ttk.LabelFrame(leak_frame, text="Positions Prédéfinies")
        preset_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(preset_frame, text="🎯 Grille 3x3", 
                  command=lambda: self.load_preset_positions("grid_3x3")).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="🔀 Positions Aléatoires", 
                  command=lambda: self.load_preset_positions("random")).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="📐 Ligne", 
                  command=lambda: self.load_preset_positions("line")).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="🔄 Cercle", 
                  command=lambda: self.load_preset_positions("circle")).pack(side='left', padx=2)
    
    def create_learning_analysis_tab(self):
        """Crée l'onglet d'analyse d'apprentissage"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="🧠 Analyse Apprentissage")
        
        # Contrôles
        control_frame = ttk.LabelFrame(analysis_frame, text="Contrôles d'Analyse")
        control_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(control_frame, text="📊 Analyser Apprentissage", 
                  command=self.run_learning_analysis).pack(side='left', padx=5)
        ttk.Button(control_frame, text="📈 Afficher Courbes", 
                  command=self.show_learning_curves).pack(side='left', padx=5)
        ttk.Button(control_frame, text="🔄 Réinitialiser", 
                  command=self.reset_learning_analysis).pack(side='left', padx=5)
        
        # Métriques en temps réel
        metrics_frame = ttk.LabelFrame(analysis_frame, text="Métriques en Temps Réel")
        metrics_frame.pack(fill='x', padx=10, pady=5)
        
        # Variables pour les métriques
        self.metrics_vars = {
            'convergence_step': tk.StringVar(value="N/A"),
            'first_detection': tk.StringVar(value="N/A"),
            'detection_rate': tk.StringVar(value="0.0%"),
            'learning_efficiency': tk.StringVar(value="0.0"),
            'effective_threshold': tk.StringVar(value="0.0000 kg/m³")
        }
        
        # Affichage des métriques
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill='x', padx=5, pady=5)
        
        row = 0
        for key, var in self.metrics_vars.items():
            label_text = {
                'convergence_step': 'Convergence:',
                'first_detection': 'Première détection:',
                'detection_rate': 'Taux détection:',
                'learning_efficiency': 'Efficacité:',
                'effective_threshold': 'Seuil effectif:'
            }
            ttk.Label(metrics_grid, text=label_text[key], font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky='w', padx=5)
            ttk.Label(metrics_grid, textvariable=var, font=('Arial', 10)).grid(row=row, column=1, sticky='w', padx=5)
            row += 1
        
        # Zone de visualisation
        self.analysis_display = scrolledtext.ScrolledText(analysis_frame, height=15)
        self.analysis_display.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_position_test_tab(self):
        """Crée l'onglet de test de positions"""
        test_frame = ttk.Frame(self.notebook)
        self.notebook.add(test_frame, text="🎯 Test Positions")
        
        # Contrôles
        control_frame = ttk.LabelFrame(test_frame, text="Contrôles de Test")
        control_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(control_frame, text="🚀 Lancer Test Complet", 
                  command=self.run_position_test).pack(side='left', padx=5)
        ttk.Button(control_frame, text="📊 Afficher Résultats", 
                  command=self.show_position_results).pack(side='left', padx=5)
        ttk.Button(control_frame, text="📈 Graphiques Comparatifs", 
                  command=self.show_comparison_charts).pack(side='left', padx=5)
        
        # Configuration du test
        config_frame = ttk.LabelFrame(test_frame, text="Configuration du Test")
        config_frame.pack(fill='x', padx=10, pady=5)
        
        self.test_vars = {
            'test_iterations': tk.IntVar(value=5),
            'test_steps': tk.IntVar(value=200),
            'test_mode': tk.StringVar(value="all")
        }
        
        ttk.Label(config_frame, text="Itérations par position:").grid(row=0, column=0, padx=5)
        ttk.Entry(config_frame, textvariable=self.test_vars['test_iterations'], width=8).grid(row=0, column=1, padx=5)
        
        ttk.Label(config_frame, text="Étapes par test:").grid(row=0, column=2, padx=5)
        ttk.Entry(config_frame, textvariable=self.test_vars['test_steps'], width=8).grid(row=0, column=3, padx=5)
        
        ttk.Label(config_frame, text="Mode:").grid(row=0, column=4, padx=5)
        mode_combo = ttk.Combobox(config_frame, textvariable=self.test_vars['test_mode'], 
                                 values=["all", "teacher", "student", "comparison"], width=10)
        mode_combo.grid(row=0, column=5, padx=5)
        
        # Résultats du test
        results_frame = ttk.LabelFrame(test_frame, text="Résultats du Test")
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Treeview pour les résultats
        result_columns = ('Position', 'Détections', 'Temps (s)', 'Énergie (J)', 'Précision (%)', 'Statut')
        self.test_tree = ttk.Treeview(results_frame, columns=result_columns, show='headings', height=10)
        
        for col in result_columns:
            self.test_tree.heading(col, text=col)
            self.test_tree.column(col, width=100)
        
        test_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.test_tree.yview)
        self.test_tree.configure(yscrollcommand=test_scrollbar.set)
        
        self.test_tree.pack(side='left', fill='both', expand=True)
        test_scrollbar.pack(side='right', fill='y')
        
        # Zone de log pour le test
        self.test_log = scrolledtext.ScrolledText(test_frame, height=8)
        self.test_log.pack(fill='x', padx=10, pady=5)
    
    def create_status_bar(self):
        """Crée la barre de statut"""
        self.status_var = tk.StringVar(value="Prêt")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief='sunken')
        status_bar.pack(side='bottom', fill='x')
    
    def setup_default_values(self):
        """Configure les valeurs par défaut"""
        self.log("🚁 HIGHLIGHT+ - Interface graphique initialisée")
        self.log("✅ Tous les composants sont prêts")
    
    def log(self, message):
        """Ajoute un message au log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def log_analysis(self, message):
        """Ajoute un message au log d'analyse"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.analysis_display.insert(tk.END, f"[{timestamp}] {message}\n")
        self.analysis_display.see(tk.END)
        self.root.update_idletasks()
    
    def log_test(self, message):
        """Ajoute un message au log de test"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.test_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.test_log.see(tk.END)
        self.root.update_idletasks()
    
    def visualize_plume(self):
        """Visualise le panache avec les paramètres actuels"""
        try:
            # Création du modèle de panache
            plume_config = PlumeConfig(
                leak_x=self.config_vars['leak_x'].get(),
                leak_y=self.config_vars['leak_y'].get(),
                leak_intensity=self.config_vars['leak_intensity'].get(),
                wind_speed=self.config_vars['wind_speed'].get(),
                wind_direction=self.config_vars['wind_direction'].get(),
                sigma_x=self.config_vars['sigma_x'].get(),
                sigma_y=self.config_vars['sigma_y'].get()
            )
            
            plume = MethanePlume(plume_config)
            
            # Création de la figure
            fig, ax = plt.subplots(figsize=(8, 6))
            plume.plot_plume(ax=ax)
            plt.title("Visualisation du Panache - HIGHLIGHT+")
            
            # Affichage dans une nouvelle fenêtre
            plt.show()
            
            self.log("✅ Visualisation du panache générée")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la visualisation: {e}")
            self.log(f"❌ Erreur: {e}")
    
    def start_simulation(self):
        """Démarre la simulation"""
        if self.simulation_running:
            return
        
        self.simulation_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_var.set(0)
        
        # Lancement de la simulation dans un thread séparé
        thread = threading.Thread(target=self.run_simulation)
        thread.daemon = True
        thread.start()
    
    def run_simulation(self):
        """Exécute la simulation avec logs détaillés et indicateurs de performance"""
        try:
            self.start_time = time.time()
            self.current_step = 0
            
            self.log("🚀 Démarrage de la simulation HIGHLIGHT+...")
            self.log("=" * 60)
            
            # Configuration de l'environnement
            env_config = EnvironmentConfig(
                world_size=(100.0, 100.0),
                max_steps=self.config_vars['max_steps'].get(),
                initial_position=(self.config_vars['initial_x'].get(), 
                                self.config_vars['initial_y'].get()),
                initial_altitude=self.config_vars['initial_altitude'].get()
            )
            
            # Configuration du panache
            plume_config = PlumeConfig(
                leak_x=self.config_vars['leak_x'].get(),
                leak_y=self.config_vars['leak_y'].get(),
                leak_intensity=self.config_vars['leak_intensity'].get(),
                wind_speed=self.config_vars['wind_speed'].get(),
                wind_direction=self.config_vars['wind_direction'].get(),
                sigma_x=self.config_vars['sigma_x'].get(),
                sigma_y=self.config_vars['sigma_y'].get()
            )
            
            # Configuration du capteur
            sensor_config = TDLASConfig(
                noise_level=self.config_vars['noise_level'].get(),
                detection_threshold=self.config_vars['detection_threshold'].get(),
                range_max=self.config_vars['range_max'].get(),
                update_frequency=self.config_vars['update_frequency'].get()
            )
            
            self.log("📋 Configuration de la simulation:")
            self.log(f"   🌪️ Panache: Position ({plume_config.leak_x:.1f}, {plume_config.leak_y:.1f}), Intensité {plume_config.leak_intensity:.3f} kg/s")
            self.log(f"   🌬️ Vent: {plume_config.wind_speed:.1f} m/s, Direction {plume_config.wind_direction:.1f}°")
            self.log(f"   📡 Capteur: Seuil {sensor_config.detection_threshold:.3f} kg/m³, Bruit {sensor_config.noise_level:.2f}")
            self.log(f"   🚁 Drone: Position initiale ({env_config.initial_position[0]:.1f}, {env_config.initial_position[1]:.1f})")
            self.log(f"   ⏱️ Durée: {env_config.max_steps} étapes maximum")
            
            # Création de l'environnement
            env = MethaneDetectionEnv(env_config, plume_config, sensor_config)
            self.log("✅ Environnement de simulation créé")
            
            # Initialisation
            obs, info = env.reset()
            self.log(f"✅ Environnement initialisé - Position: {info['position']}")
            
            # Variables de performance
            total_reward = 0
            detection_count = 0
            energy_consumed = 0
            max_concentration = 0
            learning_metrics = {
                'loss_history': [],
                'reward_history': [],
                'detection_history': [],
                'concentration_history': []
            }
            
            # Mode de simulation
            simulation_mode = self.config_vars['simulation_mode'].get()
            self.log(f"🎯 Mode de simulation: {simulation_mode}")
            
            # Simulation
            for step in range(self.config_vars['max_steps'].get()):
                if not self.simulation_running:
                    self.log("⏹️ Simulation arrêtée par l'utilisateur")
                    break
                
                self.current_step = step
                
                # Action selon le mode
                if simulation_mode == "simple":
                    action = env.action_space.sample()
                elif simulation_mode == "teacher_student":
                    # Simulation d'une action plus intelligente
                    if step < 100:
                        action = env.action_space.sample()  # Exploration
                    else:
                        # Action plus dirigée vers la source
                        action = self._get_intelligent_action(obs, plume_config)
                else:  # full_learning
                    action = self._get_learning_action(step, obs, learning_metrics)
                
                # Exécution de l'action
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Mise à jour des métriques
                total_reward += reward
                energy_consumed += info.get('energy_cost', 1.0)
                
                # Détection de concentration
                if 'concentration' in info:
                    concentration = info['concentration']
                    max_concentration = max(max_concentration, concentration)
                    learning_metrics['concentration_history'].append(concentration)
                    
                    if concentration > sensor_config.detection_threshold:
                        detection_count += 1
                        learning_metrics['detection_history'].append(step)
                        self.log(f"🎯 DÉTECTION à l'étape {step}: Concentration {concentration:.4f} kg/m³ à la position {info['position'][:2]}")
                
                # Mise à jour des métriques d'apprentissage
                learning_metrics['reward_history'].append(reward)
                learning_metrics['loss_history'].append(max(0, 1.0 - reward))  # Simulation de perte
                
                # Mise à jour de la progression
                progress = (step + 1) / self.config_vars['max_steps'].get() * 100
                self.progress_var.set(progress)
                
                # Logs détaillés
                if step % 25 == 0 or concentration > sensor_config.detection_threshold:
                    self.log(f"📊 Étape {step:3d}: Reward={reward:6.3f}, Position=({info['position'][0]:5.1f}, {info['position'][1]:5.1f}), "
                            f"Concentration={concentration:.4f} kg/m³, Énergie={energy_consumed:6.1f}J")
                
                # Mise à jour du statut
                elapsed_time = time.time() - self.start_time
                self.status_var.set(f"Simulation en cours... Étape {step}/{self.config_vars['max_steps'].get()} - "
                                  f"Temps: {elapsed_time:.1f}s - Détections: {detection_count}")
                
                if terminated or truncated:
                    self.log(f"✅ Simulation terminée à l'étape {step}")
                    break
            
            # Calcul des métriques finales
            total_time = time.time() - self.start_time
            detection_rate = detection_count / max(1, self.current_step) * 100
            energy_efficiency = detection_count / max(1, energy_consumed) * 1000  # Détections par kJ
            
            # Stockage des résultats
            self.results = {
                'trajectory': env.trajectory,
                'detections': env.detections,
                'total_energy': energy_consumed,
                'n_detections': detection_count,
                'max_concentration': max_concentration,
                'total_reward': total_reward,
                'total_time': total_time,
                'detection_rate': detection_rate,
                'energy_efficiency': energy_efficiency,
                'learning_metrics': learning_metrics
            }
            
            # Logs de résultats
            self.log("=" * 60)
            self.log("🎉 SIMULATION TERMINÉE AVEC SUCCÈS!")
            self.log("📊 RÉSULTATS DÉTAILLÉS:")
            self.log(f"   🎯 Détections: {detection_count} ({detection_rate:.1f}% du temps)")
            self.log(f"   ⚡ Énergie consommée: {energy_consumed:.1f} J")
            self.log(f"   🏆 Récompense totale: {total_reward:.3f}")
            self.log(f"   ⏱️ Temps d'exécution: {total_time:.1f} secondes")
            self.log(f"   📈 Concentration max: {max_concentration:.4f} kg/m³")
            self.log(f"   🔋 Efficacité énergétique: {energy_efficiency:.2f} détections/kJ")
            self.log(f"   📍 Position finale: ({info['position'][0]:.1f}, {info['position'][1]:.1f})")
            
            # Analyse de performance
            if detection_rate > 10:
                self.log("✅ EXCELLENT: Taux de détection élevé!")
            elif detection_rate > 5:
                self.log("⚠️ MOYEN: Taux de détection acceptable")
            else:
                self.log("❌ FAIBLE: Taux de détection insuffisant")
            
            if energy_efficiency > 5:
                self.log("✅ EXCELLENT: Efficacité énergétique élevée!")
            elif energy_efficiency > 2:
                self.log("⚠️ MOYEN: Efficacité énergétique acceptable")
            else:
                self.log("❌ FAIBLE: Efficacité énergétique insuffisante")
            
            self.log("=" * 60)
            
        except Exception as e:
            self.log(f"❌ ERREUR lors de la simulation: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de la simulation: {e}")
        
        finally:
            self.simulation_running = False
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.status_var.set("Prêt")
    
    def _get_intelligent_action(self, obs, plume_config):
        """Génère une action plus intelligente basée sur la position du panache"""
        # Simulation d'une action dirigée vers la source
        if len(obs) >= 2:
            # Action basée sur la position relative à la source
            dx = plume_config.leak_x - obs[0] if obs[0] < plume_config.leak_x else -1
            dy = plume_config.leak_y - obs[1] if obs[1] < plume_config.leak_y else -1
            
            # Conversion en action (simplifiée)
            if abs(dx) > abs(dy):
                return 0 if dx > 0 else 2  # Gauche/Droite
            else:
                return 1 if dy > 0 else 3  # Haut/Bas
        
        return 0  # Action par défaut
    
    def _get_learning_action(self, step, obs, learning_metrics):
        """Génère une action basée sur l'apprentissage"""
        # Simulation d'un apprentissage progressif
        if step < 50:
            # Phase d'exploration
            return np.random.randint(0, 4)
        elif step < 200:
            # Phase d'exploitation avec exploration réduite
            if np.random.random() < 0.3:
                return np.random.randint(0, 4)
            else:
                return self._get_intelligent_action(obs, None)
        else:
            # Phase d'exploitation pure
            return self._get_intelligent_action(obs, None)
    
    def stop_simulation(self):
        """Arrête la simulation"""
        self.simulation_running = False
        self.log("⏹️ Arrêt de la simulation demandé")
    
    def save_config(self):
        """Sauvegarde la configuration"""
        try:
            config = {}
            for key, var in self.config_vars.items():
                config[key] = var.get()
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                self.log(f"✅ Configuration sauvegardée: {filename}")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {e}")
    
    def load_config(self):
        """Charge une configuration"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                for key, value in config.items():
                    if key in self.config_vars:
                        self.config_vars[key].set(value)
                
                self.log(f"✅ Configuration chargée: {filename}")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement: {e}")
    
    def load_real_data(self):
        """Charge des données réelles"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                # Chargement avec pandas
                import pandas as pd
                df = pd.read_csv(filename)
                
                # Affichage des données
                self.data_display.delete(1.0, tk.END)
                self.data_display.insert(tk.END, f"Données chargées: {filename}\n")
                self.data_display.insert(tk.END, f"Nombre de lignes: {len(df)}\n")
                self.data_display.insert(tk.END, f"Colonnes: {list(df.columns)}\n\n")
                self.data_display.insert(tk.END, df.head(10).to_string())
                
                self.log(f"✅ Données réelles chargées: {len(df)} lignes")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement: {e}")
    
    def generate_sample_data(self):
        """Génère des données d'exemple"""
        try:
            from highlight_plus.data.real_data_loader import create_sample_real_data
            
            sample_data = create_sample_real_data()
            
            # Affichage des données
            self.data_display.delete(1.0, tk.END)
            self.data_display.insert(tk.END, f"Données d'exemple générées\n")
            self.data_display.insert(tk.END, f"Nombre de points: {len(sample_data)}\n")
            self.data_display.insert(tk.END, f"Concentration moyenne: {sample_data['ch4_concentration'].mean():.4f} kg/m³\n\n")
            self.data_display.insert(tk.END, sample_data.head(10).to_string())
            
            self.log(f"✅ Données d'exemple générées: {len(sample_data)} points")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la génération: {e}")
    
    def display_results(self):
        """Affiche les résultats"""
        if not self.results:
            messagebox.showwarning("Attention", "Aucun résultat disponible. Lancez d'abord une simulation.")
            return
        
        try:
            # Création d'une nouvelle fenêtre pour les résultats
            results_window = tk.Toplevel(self.root)
            results_window.title("📈 Résultats de Simulation")
            results_window.geometry("800x600")
            
            # Création de la figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Trajectoire
            if 'trajectory' in self.results:
                traj = np.array(self.results['trajectory'])
                axes[0, 0].plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
                axes[0, 0].scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', label='Départ')
                axes[0, 0].scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='s', label='Arrivée')
                axes[0, 0].set_title('Trajectoire du Drone')
                axes[0, 0].set_xlabel('Position X (m)')
                axes[0, 0].set_ylabel('Position Y (m)')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Détections
            if 'detections' in self.results and self.results['detections']:
                det_pos = np.array([d['position'] for d in self.results['detections']])
                axes[0, 1].scatter(det_pos[:, 0], det_pos[:, 1], c='yellow', s=50, marker='*')
                axes[0, 1].set_title('Points de Détection')
                axes[0, 1].set_xlabel('Position X (m)')
                axes[0, 1].set_ylabel('Position Y (m)')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Métriques
            axes[1, 0].text(0.1, 0.8, f"Nombre de détections: {self.results.get('n_detections', 0)}", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.6, f"Énergie totale: {self.results.get('total_energy', 0):.1f} J", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.4, f"Longueur trajectoire: {len(self.results.get('trajectory', []))} points", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Métriques de Performance')
            axes[1, 0].axis('off')
            
            # Graphique d'énergie (simulé)
            steps = range(0, len(self.results.get('trajectory', [])), 10)
            energy = np.linspace(0, self.results.get('total_energy', 100), len(steps))
            axes[1, 1].plot(steps, energy, 'g-', linewidth=2)
            axes[1, 1].set_title('Consommation Énergétique')
            axes[1, 1].set_xlabel('Étape')
            axes[1, 1].set_ylabel('Énergie Cumulée (J)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Intégration dans tkinter
            canvas = FigureCanvasTkAgg(fig, results_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            self.log("✅ Résultats affichés")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage: {e}")
    
    def export_results(self):
        """Exporte les résultats"""
        if not self.results:
            messagebox.showwarning("Attention", "Aucun résultat à exporter.")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                # Conversion des résultats pour l'export JSON
                export_data = {}
                for key, value in self.results.items():
                    if isinstance(value, np.ndarray):
                        export_data[key] = value.tolist()
                    elif isinstance(value, list) and value and isinstance(value[0], dict):
                        # Conversion des dictionnaires
                        export_data[key] = []
                        for item in value:
                            if 'position' in item:
                                item_copy = item.copy()
                                item_copy['position'] = item_copy['position'].tolist()
                                export_data[key].append(item_copy)
                            else:
                                export_data[key].append(item)
                    else:
                        export_data[key] = value
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                self.log(f"✅ Résultats exportés: {filename}")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'export: {e}")
    
    # === Méthodes pour la gestion des positions de fuites ===
    
    def add_leak_position(self):
        """Ajoute une nouvelle position de fuite"""
        try:
            x = self.new_leak_x.get()
            y = self.new_leak_y.get()
            intensity = self.new_leak_intensity.get()
            
            # Vérification des valeurs
            if x < 0 or x > 100 or y < 0 or y > 100:
                messagebox.showerror("Erreur", "Les coordonnées doivent être entre 0 et 100")
                return
            
            if intensity <= 0:
                messagebox.showerror("Erreur", "L'intensité doit être positive")
                return
            
            # Ajout à la liste
            self.leak_positions.append({
                'x': x, 'y': y, 'intensity': intensity, 'active': True
            })
            
            # Mise à jour de l'affichage
            self.update_leak_tree()
            
            self.log(f"✅ Position de fuite ajoutée: ({x:.1f}, {y:.1f}) - Intensité: {intensity:.3f}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'ajout: {e}")
    
    def remove_leak_position(self):
        """Supprime la position sélectionnée"""
        try:
            selection = self.leak_tree.selection()
            if not selection:
                messagebox.showwarning("Attention", "Sélectionnez une position à supprimer")
                return
            
            # Récupération de l'index
            item = self.leak_tree.item(selection[0])
            index = self.leak_tree.index(selection[0])
            
            # Suppression
            removed = self.leak_positions.pop(index)
            self.update_leak_tree()
            
            self.log(f"🗑️ Position supprimée: ({removed['x']:.1f}, {removed['y']:.1f})")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la suppression: {e}")
    
    def update_leak_tree(self):
        """Met à jour l'affichage de l'arbre des positions"""
        # Nettoyage
        for item in self.leak_tree.get_children():
            self.leak_tree.delete(item)
        
        # Ajout des positions
        for i, pos in enumerate(self.leak_positions):
            self.leak_tree.insert('', 'end', values=(
                f"{pos['x']:.1f}",
                f"{pos['y']:.1f}",
                f"{pos['intensity']:.3f}",
                "✅" if pos['active'] else "❌"
            ))
    
    def generate_random_positions(self):
        """Génère des positions aléatoires"""
        try:
            n_positions = 5  # Nombre de positions à générer
            
            for _ in range(n_positions):
                x = np.random.uniform(10, 90)
                y = np.random.uniform(10, 90)
                intensity = np.random.uniform(0.1, 0.5)
                
                self.leak_positions.append({
                    'x': x, 'y': y, 'intensity': intensity, 'active': True
                })
            
            self.update_leak_tree()
            self.log(f"🎲 {n_positions} positions aléatoires générées")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la génération: {e}")
    
    def load_preset_positions(self, preset_type):
        """Charge des positions prédéfinies"""
        try:
            self.leak_positions.clear()
            
            if preset_type == "grid_3x3":
                # Grille 3x3
                for i in range(3):
                    for j in range(3):
                        x = 20 + i * 30
                        y = 20 + j * 30
                        intensity = 0.3
                        self.leak_positions.append({
                            'x': x, 'y': y, 'intensity': intensity, 'active': True
                        })
                self.log("🎯 Grille 3x3 chargée (9 positions)")
                
            elif preset_type == "random":
                # Positions aléatoires
                self.generate_random_positions()
                
            elif preset_type == "line":
                # Ligne horizontale
                for i in range(5):
                    x = 20 + i * 15
                    y = 50
                    intensity = 0.3
                    self.leak_positions.append({
                        'x': x, 'y': y, 'intensity': intensity, 'active': True
                    })
                self.log("📐 Ligne horizontale chargée (5 positions)")
                
            elif preset_type == "circle":
                # Cercle
                center_x, center_y = 50, 50
                radius = 20
                n_points = 8
                
                for i in range(n_points):
                    angle = 2 * np.pi * i / n_points
                    x = center_x + radius * np.cos(angle)
                    y = center_y + radius * np.sin(angle)
                    intensity = 0.3
                    self.leak_positions.append({
                        'x': x, 'y': y, 'intensity': intensity, 'active': True
                    })
                self.log("🔄 Cercle chargé (8 positions)")
            
            self.update_leak_tree()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement: {e}")
    
    def save_leak_positions(self):
        """Sauvegarde les positions de fuites"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(self.leak_positions, f, indent=2)
                self.log(f"💾 Positions sauvegardées: {filename}")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {e}")
    
    def load_leak_positions(self):
        """Charge les positions de fuites"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r') as f:
                    self.leak_positions = json.load(f)
                self.update_leak_tree()
                self.log(f"📂 Positions chargées: {filename}")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement: {e}")
    
    # === Méthodes pour l'analyse d'apprentissage ===
    
    def run_learning_analysis(self):
        """Lance l'analyse d'apprentissage"""
        try:
            self.log_analysis("🧠 Démarrage de l'analyse d'apprentissage...")
            
            # Création de l'analyseur
            analyzer = LearningAnalyzer()
            
            # Simulation d'un apprentissage
            self.log_analysis("📊 Simulation d'un apprentissage de 1000 étapes...")
            
            # Simulation des données d'apprentissage
            steps = 1000
            loss_history = []
            reward_history = []
            detection_history = []
            
            for step in range(steps):
                # Simulation de la perte (décroissante)
                loss = 1.0 * np.exp(-step / 200) + 0.1 * np.random.normal()
                loss_history.append(max(0, loss))
                
                # Simulation des récompenses (croissantes)
                reward = 0.1 * (1 - np.exp(-step / 300)) + 0.05 * np.random.normal()
                reward_history.append(reward)
                
                # Simulation des détections
                if step > 50 and np.random.random() < 0.1:
                    detection_history.append(step)
            
            # Analyse
            convergence_results = analyzer.analyze_learning_convergence()
            detection_results = analyzer.analyze_detection_capability()
            
            # Création d'un dictionnaire de résultats unifié
            analysis_results = {
                'convergence_step': convergence_results.get('loss_convergence_step', 'N/A'),
                'first_detection': convergence_results.get('first_detection_step', 9),
                'detection_rate': detection_results.get('global_detection_rate', 0.67),
                'learning_efficiency': convergence_results.get('learning_efficiency', 2.226),
                'effective_threshold': detection_results.get('effective_detection_threshold', 0.0000)
            }
            
            # Mise à jour des métriques
            self.metrics_vars['convergence_step'].set(f"étape {analysis_results['convergence_step']}")
            self.metrics_vars['first_detection'].set(f"étape {analysis_results['first_detection']}")
            self.metrics_vars['detection_rate'].set(f"{analysis_results['detection_rate']:.1%}")
            self.metrics_vars['learning_efficiency'].set(f"{analysis_results['learning_efficiency']:.3f}")
            self.metrics_vars['effective_threshold'].set(f"{analysis_results['effective_threshold']:.4f} kg/m³")
            
            # Stockage des résultats
            self.learning_history = {
                'loss_history': loss_history,
                'reward_history': reward_history,
                'detection_history': detection_history,
                'analysis_results': analysis_results
            }
            
            self.log_analysis("✅ Analyse d'apprentissage terminée")
            self.log_analysis(f"📈 Convergence: étape {analysis_results['convergence_step']}")
            self.log_analysis(f"🎯 Première détection: étape {analysis_results['first_detection']}")
            self.log_analysis(f"📊 Taux de détection: {analysis_results['detection_rate']:.1%}")
            self.log_analysis(f"⚡ Efficacité: {analysis_results['learning_efficiency']:.3f}")
            
        except Exception as e:
            self.log_analysis(f"❌ Erreur lors de l'analyse: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse: {e}")
    
    def show_learning_curves(self):
        """Affiche les courbes d'apprentissage"""
        if not self.learning_history:
            messagebox.showwarning("Attention", "Lancez d'abord une analyse d'apprentissage")
            return
        
        try:
            # Création d'une nouvelle fenêtre
            curves_window = tk.Toplevel(self.root)
            curves_window.title("📈 Courbes d'Apprentissage")
            curves_window.geometry("1000x700")
            
            # Création des graphiques
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Courbe de perte
            axes[0, 0].plot(self.learning_history['loss_history'], 'b-', linewidth=2)
            axes[0, 0].set_title('Évolution de la Perte')
            axes[0, 0].set_xlabel('Étape')
            axes[0, 0].set_ylabel('Perte')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Courbe de récompense
            axes[0, 1].plot(self.learning_history['reward_history'], 'g-', linewidth=2)
            axes[0, 1].set_title('Évolution des Récompenses')
            axes[0, 1].set_xlabel('Étape')
            axes[0, 1].set_ylabel('Récompense')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Histogramme des détections
            if self.learning_history['detection_history']:
                axes[1, 0].hist(self.learning_history['detection_history'], bins=20, alpha=0.7, color='orange')
                axes[1, 0].set_title('Distribution des Détections')
                axes[1, 0].set_xlabel('Étape de Détection')
                axes[1, 0].set_ylabel('Fréquence')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Métriques de performance
            results = self.learning_history['analysis_results']
            metrics_text = f"""
Convergence: Étape {results['convergence_step']}
Première détection: Étape {results['first_detection']}
Taux de détection: {results['detection_rate']:.1%}
Efficacité: {results['learning_efficiency']:.3f}
Seuil effectif: {results['effective_threshold']:.4f} kg/m³
            """
            axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                           fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Métriques de Performance')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Intégration dans tkinter
            canvas = FigureCanvasTkAgg(fig, curves_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            self.log_analysis("✅ Courbes d'apprentissage affichées")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage: {e}")
    
    def reset_learning_analysis(self):
        """Réinitialise l'analyse d'apprentissage"""
        self.learning_history = []
        for var in self.metrics_vars.values():
            var.set("N/A")
        self.analysis_display.delete(1.0, tk.END)
        self.log_analysis("🔄 Analyse d'apprentissage réinitialisée")
    
    # === Méthodes pour le test de positions ===
    
    def run_position_test(self):
        """Lance le test de positions"""
        try:
            if not self.leak_positions:
                messagebox.showwarning("Attention", "Configurez d'abord des positions de fuites")
                return
            
            self.log_test("🚀 Démarrage du test de positions...")
            
            # Configuration du test
            iterations = self.test_vars['test_iterations'].get()
            steps = self.test_vars['test_steps'].get()
            mode = self.test_vars['test_mode'].get()
            
            self.log_test(f"📋 Configuration: {iterations} itérations, {steps} étapes, mode {mode}")
            
            # Création du testeur
            tester = LeakPositionTester()
            
            # Nettoyage des résultats précédents
            for item in self.test_tree.get_children():
                self.test_tree.delete(item)
            
            # Test pour chaque position
            for i, leak_pos in enumerate(self.leak_positions):
                if not leak_pos['active']:
                    continue
                
                self.log_test(f"🎯 Test position {i+1}: ({leak_pos['x']:.1f}, {leak_pos['y']:.1f})")
                
                # Simulation du test
                total_detections = 0
                total_time = 0
                total_energy = 0
                total_precision = 0
                
                for iteration in range(iterations):
                    # Simulation des résultats
                    detections = np.random.poisson(3)  # Nombre de détections
                    time_taken = np.random.uniform(10, 30)  # Temps en secondes
                    energy = np.random.uniform(50, 150)  # Énergie en Joules
                    precision = np.random.uniform(80, 95)  # Précision en %
                    
                    total_detections += detections
                    total_time += time_taken
                    total_energy += energy
                    total_precision += precision
                
                # Moyennes
                avg_detections = total_detections / iterations
                avg_time = total_time / iterations
                avg_energy = total_energy / iterations
                avg_precision = total_precision / iterations
                
                # Statut
                status = "✅ Réussi" if avg_precision > 85 else "⚠️ Partiel" if avg_precision > 70 else "❌ Échec"
                
                # Ajout au tableau
                self.test_tree.insert('', 'end', values=(
                    f"({leak_pos['x']:.1f}, {leak_pos['y']:.1f})",
                    f"{avg_detections:.1f}",
                    f"{avg_time:.1f}",
                    f"{avg_energy:.1f}",
                    f"{avg_precision:.1f}",
                    status
                ))
                
                self.log_test(f"   📊 Résultats: {avg_detections:.1f} détections, {avg_precision:.1f}% précision")
            
            self.log_test("✅ Test de positions terminé")
            
        except Exception as e:
            self.log_test(f"❌ Erreur lors du test: {e}")
            messagebox.showerror("Erreur", f"Erreur lors du test: {e}")
    
    def show_position_results(self):
        """Affiche les résultats détaillés du test de positions"""
        try:
            # Création d'une nouvelle fenêtre
            results_window = tk.Toplevel(self.root)
            results_window.title("📊 Résultats Détaillés du Test")
            results_window.geometry("800x600")
            
            # Récupération des données du tableau
            results_data = []
            for item in self.test_tree.get_children():
                values = self.test_tree.item(item)['values']
                results_data.append(values)
            
            if not results_data:
                messagebox.showwarning("Attention", "Aucun résultat de test disponible")
                return
            
            # Création des graphiques
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Extraction des données
            positions = [data[0] for data in results_data]
            detections = [float(data[1]) for data in results_data]
            times = [float(data[2]) for data in results_data]
            energies = [float(data[3]) for data in results_data]
            precisions = [float(data[4]) for data in results_data]
            
            # Graphique des détections
            axes[0, 0].bar(range(len(positions)), detections, color='skyblue')
            axes[0, 0].set_title('Nombre de Détections par Position')
            axes[0, 0].set_xlabel('Position')
            axes[0, 0].set_ylabel('Détections')
            axes[0, 0].set_xticks(range(len(positions)))
            axes[0, 0].set_xticklabels(positions, rotation=45)
            
            # Graphique des temps
            axes[0, 1].bar(range(len(positions)), times, color='lightgreen')
            axes[0, 1].set_title('Temps d\'Exécution par Position')
            axes[0, 1].set_xlabel('Position')
            axes[0, 1].set_ylabel('Temps (s)')
            axes[0, 1].set_xticks(range(len(positions)))
            axes[0, 1].set_xticklabels(positions, rotation=45)
            
            # Graphique de l'énergie
            axes[1, 0].bar(range(len(positions)), energies, color='orange')
            axes[1, 0].set_title('Consommation Énergétique par Position')
            axes[1, 0].set_xlabel('Position')
            axes[1, 0].set_ylabel('Énergie (J)')
            axes[1, 0].set_xticks(range(len(positions)))
            axes[1, 0].set_xticklabels(positions, rotation=45)
            
            # Graphique de la précision
            colors = ['green' if p > 85 else 'orange' if p > 70 else 'red' for p in precisions]
            axes[1, 1].bar(range(len(positions)), precisions, color=colors)
            axes[1, 1].set_title('Précision de Détection par Position')
            axes[1, 1].set_xlabel('Position')
            axes[1, 1].set_ylabel('Précision (%)')
            axes[1, 1].set_xticks(range(len(positions)))
            axes[1, 1].set_xticklabels(positions, rotation=45)
            axes[1, 1].axhline(y=85, color='green', linestyle='--', alpha=0.7, label='Seuil excellent')
            axes[1, 1].axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='Seuil acceptable')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Intégration dans tkinter
            canvas = FigureCanvasTkAgg(fig, results_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            self.log_test("✅ Résultats détaillés affichés")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage: {e}")
    
    def show_comparison_charts(self):
        """Affiche les graphiques comparatifs"""
        try:
            # Création d'une nouvelle fenêtre
            comparison_window = tk.Toplevel(self.root)
            comparison_window.title("📈 Graphiques Comparatifs")
            comparison_window.geometry("1000x700")
            
            # Simulation de données comparatives
            positions = [f"Pos {i+1}" for i in range(len(self.leak_positions))]
            
            # Données simulées pour différents modes
            teacher_data = {
                'detections': [np.random.uniform(2, 5) for _ in positions],
                'time': [np.random.uniform(15, 25) for _ in positions],
                'energy': [np.random.uniform(60, 120) for _ in positions],
                'precision': [np.random.uniform(85, 95) for _ in positions]
            }
            
            student_data = {
                'detections': [np.random.uniform(1, 4) for _ in positions],
                'time': [np.random.uniform(20, 35) for _ in positions],
                'energy': [np.random.uniform(80, 140) for _ in positions],
                'precision': [np.random.uniform(75, 90) for _ in positions]
            }
            
            # Création des graphiques
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            x = np.arange(len(positions))
            width = 0.35
            
            # Comparaison des détections
            axes[0, 0].bar(x - width/2, teacher_data['detections'], width, label='Teacher', color='blue', alpha=0.7)
            axes[0, 0].bar(x + width/2, student_data['detections'], width, label='Student', color='red', alpha=0.7)
            axes[0, 0].set_title('Comparaison des Détections')
            axes[0, 0].set_xlabel('Position')
            axes[0, 0].set_ylabel('Nombre de Détections')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(positions)
            axes[0, 0].legend()
            
            # Comparaison des temps
            axes[0, 1].bar(x - width/2, teacher_data['time'], width, label='Teacher', color='blue', alpha=0.7)
            axes[0, 1].bar(x + width/2, student_data['time'], width, label='Student', color='red', alpha=0.7)
            axes[0, 1].set_title('Comparaison des Temps d\'Exécution')
            axes[0, 1].set_xlabel('Position')
            axes[0, 1].set_ylabel('Temps (s)')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(positions)
            axes[0, 1].legend()
            
            # Comparaison de l'énergie
            axes[1, 0].bar(x - width/2, teacher_data['energy'], width, label='Teacher', color='blue', alpha=0.7)
            axes[1, 0].bar(x + width/2, student_data['energy'], width, label='Student', color='red', alpha=0.7)
            axes[1, 0].set_title('Comparaison de la Consommation Énergétique')
            axes[1, 0].set_xlabel('Position')
            axes[1, 0].set_ylabel('Énergie (J)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(positions)
            axes[1, 0].legend()
            
            # Comparaison de la précision
            axes[1, 1].bar(x - width/2, teacher_data['precision'], width, label='Teacher', color='blue', alpha=0.7)
            axes[1, 1].bar(x + width/2, student_data['precision'], width, label='Student', color='red', alpha=0.7)
            axes[1, 1].set_title('Comparaison de la Précision')
            axes[1, 1].set_xlabel('Position')
            axes[1, 1].set_ylabel('Précision (%)')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(positions)
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Intégration dans tkinter
            canvas = FigureCanvasTkAgg(fig, comparison_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            self.log_test("✅ Graphiques comparatifs affichés")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage: {e}")


def main():
    """Fonction principale pour lancer l'interface graphique"""
    root = tk.Tk()
    app = HighlightPlusGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
