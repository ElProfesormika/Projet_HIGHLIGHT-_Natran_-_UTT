"""
Outils de visualisation pour HIGHLIGHT+
Visualisations avancées pour l'analyse des performances et des trajectoires
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


@dataclass
class PlotConfig:
    """Configuration pour les visualisations"""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = "seaborn-v0_8"
    color_palette: str = "viridis"
    save_format: str = "png"
    save_dpi: int = 300


class HighlightPlotter:
    """
    Classe principale pour les visualisations HIGHLIGHT+
    
    Fournit des méthodes pour visualiser :
    - Trajectoires des agents (Teacher vs Student)
    - Cartes de concentration et d'incertitude
    - Métriques de performance
    - Comparaisons A/B
    - Animations des missions
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self._setup_style()
    
    def _setup_style(self):
        """Configure le style des graphiques"""
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
        
    def plot_trajectory_comparison(self, teacher_trajectory: List[Tuple[float, float]],
                                 student_trajectory: List[Tuple[float, float]],
                                 plume_data: Optional[Dict] = None,
                                 detections: Optional[List[Dict]] = None,
                                 title: str = "Comparaison des Trajectoires",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare les trajectoires du Teacher et du Student
        
        Args:
            teacher_trajectory: Trajectoire de l'Expert
            student_trajectory: Trajectoire de l'Apprenti
            plume_data: Données du panache (X, Y, C)
            detections: Liste des détections
            title: Titre du graphique
            save_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Affichage du panache
        if plume_data is not None:
            X, Y, C = plume_data['X'], plume_data['Y'], plume_data['C']
            im = ax.contourf(X, Y, C, levels=20, cmap='viridis', alpha=0.6)
            ax.contour(X, Y, C, levels=10, colors='black', alpha=0.3, linewidths=0.5)
            
            # Barre de couleur
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Concentration (kg/m³)')
        
        # Trajectoire du Teacher
        if teacher_trajectory:
            teacher_x = [pos[0] for pos in teacher_trajectory]
            teacher_y = [pos[1] for pos in teacher_trajectory]
            ax.plot(teacher_x, teacher_y, 'r-', linewidth=3, label='Expert (Teacher)', alpha=0.8)
            ax.scatter(teacher_x[0], teacher_y[0], c='green', s=100, marker='o', 
                      label='Départ Teacher', zorder=5)
            ax.scatter(teacher_x[-1], teacher_y[-1], c='red', s=100, marker='s', 
                      label='Arrivée Teacher', zorder=5)
        
        # Trajectoire du Student
        if student_trajectory:
            student_x = [pos[0] for pos in student_trajectory]
            student_y = [pos[1] for pos in student_trajectory]
            ax.plot(student_x, student_y, 'b--', linewidth=2, label='Apprenti (Student)', alpha=0.8)
            ax.scatter(student_x[0], student_y[0], c='lightgreen', s=100, marker='o', 
                      label='Départ Student', zorder=5)
            ax.scatter(student_x[-1], student_y[-1], c='blue', s=100, marker='s', 
                      label='Arrivée Student', zorder=5)
        
        # Détections
        if detections:
            for detection in detections:
                pos = detection['position']
                ax.scatter(pos[0], pos[1], c='yellow', s=80, marker='*', 
                          label='Détection' if detection == detections[0] else "", zorder=6)
        
        # Configuration
        ax.set_xlabel('Position X (m)')
        ax.set_ylabel('Position Y (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, bbox_inches='tight')
        
        return fig
    
    def plot_performance_metrics(self, metrics_data: Dict[str, List[float]],
                               title: str = "Métriques de Performance",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualise les métriques de performance
        
        Args:
            metrics_data: Dictionnaire avec les métriques
            title: Titre du graphique
            save_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Figure matplotlib
        """
        n_metrics = len(metrics_data)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_metrics))
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[i] if i < 4 else axes[-1]
            
            ax.plot(values, color=colors[i], linewidth=2, label=metric_name)
            ax.set_xlabel('Épisode')
            ax.set_ylabel(metric_name)
            ax.set_title(f'Évolution de {metric_name}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Masquer les axes inutilisés
        for i in range(len(metrics_data), 4):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_evolution(self, uncertainty_history: List[float],
                                 title: str = "Évolution de l'Incertitude",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualise l'évolution de l'incertitude du modèle
        
        Args:
            uncertainty_history: Historique de l'incertitude
            title: Titre du graphique
            save_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Figure matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Évolution temporelle
        ax1.plot(uncertainty_history, 'b-', linewidth=2)
        ax1.set_xlabel('Étape')
        ax1.set_ylabel('Incertitude')
        ax1.set_title('Évolution de l\'Incertitude')
        ax1.grid(True, alpha=0.3)
        
        # Distribution
        ax2.hist(uncertainty_history, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Incertitude')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution de l\'Incertitude')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, bbox_inches='tight')
        
        return fig
    
    def plot_energy_consumption(self, teacher_energy: List[float],
                              student_energy: List[float],
                              title: str = "Consommation Énergétique",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare la consommation énergétique des deux approches
        
        Args:
            teacher_energy: Consommation du Teacher
            student_energy: Consommation du Student
            title: Titre du graphique
            save_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Figure matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Comparaison cumulative
        teacher_cumulative = np.cumsum(teacher_energy)
        student_cumulative = np.cumsum(student_energy)
        
        ax1.plot(teacher_cumulative, 'r-', linewidth=2, label='Expert (Teacher)')
        ax1.plot(student_cumulative, 'b-', linewidth=2, label='Apprenti (Student)')
        ax1.set_xlabel('Étape')
        ax1.set_ylabel('Énergie Cumulée (J)')
        ax1.set_title('Consommation Énergétique Cumulée')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Comparaison instantanée
        ax2.plot(teacher_energy, 'r-', linewidth=2, label='Expert (Teacher)', alpha=0.7)
        ax2.plot(student_energy, 'b-', linewidth=2, label='Apprenti (Student)', alpha=0.7)
        ax2.set_xlabel('Étape')
        ax2.set_ylabel('Puissance (W)')
        ax2.set_title('Consommation Énergétique Instantanée')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, bbox_inches='tight')
        
        return fig
    
    def create_3d_visualization(self, trajectory: List[Tuple[float, float, float]],
                              plume_data: Optional[Dict] = None,
                              title: str = "Visualisation 3D de la Mission",
                              save_path: Optional[str] = None) -> go.Figure:
        """
        Crée une visualisation 3D interactive avec Plotly
        
        Args:
            trajectory: Trajectoire 3D
            plume_data: Données du panache
            title: Titre du graphique
            save_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Figure Plotly
        """
        fig = go.Figure()
        
        # Trajectoire
        if trajectory:
            x, y, z = zip(*trajectory)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                name='Trajectoire',
                line=dict(color='red', width=4),
                marker=dict(size=3, color='red')
            ))
        
        # Panache (si disponible)
        if plume_data is not None:
            X, Y, C = plume_data['X'], plume_data['Y'], plume_data['C']
            fig.add_trace(go.Surface(
                x=X, y=Y, z=C,
                colorscale='Viridis',
                opacity=0.6,
                name='Panache de méthane'
            ))
        
        # Configuration
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Position X (m)',
                yaxis_title='Position Y (m)',
                zaxis_title='Altitude (m)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_animation(self, trajectory_history: List[List[Tuple[float, float]]],
                        plume_data: Optional[Dict] = None,
                        title: str = "Animation de la Mission",
                        save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        Crée une animation de la mission
        
        Args:
            trajectory_history: Historique des trajectoires
            plume_data: Données du panache
            title: Titre de l'animation
            save_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Animation matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Initialisation
        if plume_data is not None:
            X, Y, C = plume_data['X'], plume_data['Y'], plume_data['C']
            im = ax.contourf(X, Y, C, levels=20, cmap='viridis', alpha=0.6)
            plt.colorbar(im, ax=ax, label='Concentration (kg/m³)')
        
        # Éléments animés
        line, = ax.plot([], [], 'r-', linewidth=2, label='Trajectoire')
        point, = ax.plot([], [], 'ro', markersize=8, label='Position actuelle')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Position X (m)')
        ax.set_ylabel('Position Y (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        def animate(frame):
            if frame < len(trajectory_history):
                traj = trajectory_history[frame]
                if traj:
                    x, y = zip(*traj)
                    line.set_data(x, y)
                    point.set_data([x[-1]], [y[-1]])
            return line, point
        
        anim = animation.FuncAnimation(fig, animate, frames=len(trajectory_history),
                                     interval=100, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        return anim
    
    def plot_ab_comparison(self, results_a: Dict[str, Any],
                         results_b: Dict[str, Any],
                         title: str = "Comparaison A/B",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Crée un graphique de comparaison A/B
        
        Args:
            results_a: Résultats de la méthode A (Teacher)
            results_b: Résultats de la méthode B (Student)
            title: Titre du graphique
            save_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Figure matplotlib
        """
        metrics = ['detection_rate', 'energy_efficiency', 'localization_accuracy', 'mission_time']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        values_a = [results_a.get(metric, 0) for metric in metrics]
        values_b = [results_b.get(metric, 0) for metric in metrics]
        
        bars_a = ax.bar(x - width/2, values_a, width, label='Expert (Teacher)', alpha=0.8)
        bars_b = ax.bar(x + width/2, values_b, width, label='Apprenti (Student)', alpha=0.8)
        
        # Ajout des valeurs sur les barres
        for bars in [bars_a, bars_b]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        ax.set_xlabel('Métriques')
        ax.set_ylabel('Valeurs')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.save_dpi, bbox_inches='tight')
        
        return fig


def create_test_plotter() -> HighlightPlotter:
    """Crée un plotter de test avec des paramètres par défaut"""
    config = PlotConfig(
        figure_size=(12, 8),
        style="seaborn-v0_8",
        color_palette="viridis"
    )
    return HighlightPlotter(config)


if __name__ == "__main__":
    # Test du plotter
    plotter = create_test_plotter()
    
    # Données de test
    teacher_traj = [(i, i*0.5 + np.random.normal(0, 2)) for i in range(0, 50, 2)]
    student_traj = [(i, i*0.3 + np.random.normal(0, 3)) for i in range(0, 50, 2)]
    
    # Test de visualisation
    fig = plotter.plot_trajectory_comparison(
        teacher_trajectory=teacher_traj,
        student_trajectory=student_traj,
        title="Test de Comparaison des Trajectoires"
    )
    plt.show()
    
    # Test des métriques
    metrics_data = {
        'Reward': np.random.randn(100).cumsum(),
        'Loss': np.random.exponential(0.1, 100),
        'Detection Rate': np.random.beta(2, 5, 100),
        'Energy': np.random.gamma(2, 0.5, 100)
    }
    
    fig = plotter.plot_performance_metrics(metrics_data)
    plt.show()
    
    print("Tests de visualisation terminés avec succès!")










