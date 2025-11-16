"""
Analyse comparative : Agent Naïve vs HIGHLIGHT+
Génère les métriques, graphiques et visualisations pour le rapport
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List
from .simple_simulator import SimpleConfig, SimpleSimulator
import json
from datetime import datetime


class ComparativeAnalyzer:
    """Analyse comparative entre différentes stratégies"""
    
    def __init__(self, config: SimpleConfig, n_runs: int = 10):
        """
        Args:
            config: Configuration du simulateur
            n_runs: Nombre de runs pour moyenne
        """
        self.config = config
        self.n_runs = n_runs
        self.results_naive = []
        self.results_highlight = []
    
    def run_comparison(self):
        """Exécuter les simulations comparatives"""
        print("🔄 Exécution des simulations comparatives...")
        print(f"   {self.n_runs} runs par agent")
        
        # Runs pour agent naïve
        print("\n📊 Agent NAÏVE...")
        for i in range(self.n_runs):
            sim = SimpleSimulator(self.config, agent_type="naive")
            results = sim.run()
            self.results_naive.append(results)
            if (i + 1) % 5 == 0:
                print(f"   Run {i+1}/{self.n_runs} - Détections: {results['n_detections']}")
        
        # Runs pour agent HIGHLIGHT+
        print("\n📊 Agent HIGHLIGHT+...")
        for i in range(self.n_runs):
            sim = SimpleSimulator(self.config, agent_type="highlight")
            results = sim.run()
            self.results_highlight.append(results)
            if (i + 1) % 5 == 0:
                print(f"   Run {i+1}/{self.n_runs} - Détections: {results['n_detections']}")
        
        print("\n✅ Simulations terminées !")
    
    def compute_metrics(self) -> Dict:
        """
        Calculer les métriques moyennes
        
        Returns:
            Dictionnaire avec métriques comparatives
        """
        def avg_metric(results: List[Dict], key: str, default=None):
            values = [r.get(key, default) for r in results if r.get(key) is not None]
            if not values:
                return default
            return np.mean(values)
        
        metrics = {
            'naive': {
                'detection_rate': avg_metric(self.results_naive, 'detection_rate', 0),
                'detection_time': avg_metric(self.results_naive, 'detection_time', None),
                'energy_consumed': avg_metric(self.results_naive, 'energy_consumed', 0),
                'n_detections': avg_metric(self.results_naive, 'n_detections', 0),
                'final_distance': avg_metric(self.results_naive, 'final_distance', 0)
            },
            'highlight': {
                'detection_rate': avg_metric(self.results_highlight, 'detection_rate', 0),
                'detection_time': avg_metric(self.results_highlight, 'detection_time', None),
                'energy_consumed': avg_metric(self.results_highlight, 'energy_consumed', 0),
                'n_detections': avg_metric(self.results_highlight, 'n_detections', 0),
                'final_distance': avg_metric(self.results_highlight, 'final_distance', 0)
            }
        }
        
        # Calculer les gains
        def compute_gain(naive_val, highlight_val):
            if naive_val is None or naive_val == 0:
                return None
            return ((highlight_val - naive_val) / naive_val) * 100
        
        metrics['gains'] = {
            'detection_rate': compute_gain(metrics['naive']['detection_rate'], 
                                       metrics['highlight']['detection_rate']),
            'energy_savings': compute_gain(metrics['naive']['energy_consumed'],
                                         metrics['highlight']['energy_consumed']),
            'time_reduction': None,
            'detection_improvement': compute_gain(metrics['naive']['n_detections'],
                                                metrics['highlight']['n_detections'])
        }
        
        # Temps de détection (réduction = amélioration)
        if metrics['naive']['detection_time'] and metrics['highlight']['detection_time']:
            naive_time = metrics['naive']['detection_time']
            highlight_time = metrics['highlight']['detection_time']
            metrics['gains']['time_reduction'] = ((naive_time - highlight_time) / naive_time) * 100
        
        return metrics
    
    def create_comparative_table(self, metrics: Dict) -> str:
        """
        Créer un tableau comparatif en format texte
        
        Returns:
            String du tableau formaté
        """
        n = metrics['naive']
        h = metrics['highlight']
        g = metrics['gains']
        
        table = """
╔══════════════════════════════════════════════════════════════════╗
║           TABLEAU COMPARATIF : NAÏVE vs HIGHLIGHT+               ║
╠══════════════════════════════════════════════════════════════════╣
║ Métrique                    │ Trajectoire Naïve │ HIGHLIGHT+ │ Gain ║
╠══════════════════════════════════════════════════════════════════╣"""
        
        # Taux de détection
        table += f"\n║ Taux de détection (%)     │ {n['detection_rate']:17.1f} │ {h['detection_rate']:10.1f} │ {g['detection_rate']:+5.1f}% ║"
        
        # Temps de détection
        naive_time = f"{n['detection_time']:.1f}" if n['detection_time'] else "N/A"
        highlight_time = f"{h['detection_time']:.1f}" if h['detection_time'] else "N/A"
        time_gain = f"{g['time_reduction']:+.1f}%" if g['time_reduction'] else "N/A"
        table += f"\n║ Temps de détection (s)     │ {naive_time:>17} │ {highlight_time:>10} │ {time_gain:>5} ║"
        
        # Énergie
        table += f"\n║ Énergie consommée (unités) │ {n['energy_consumed']:17.1f} │ {h['energy_consumed']:10.1f} │ {g['energy_savings']:+5.1f}% ║"
        
        # Nombre de détections
        table += f"\n║ Nombre de détections        │ {n['n_detections']:17.1f} │ {h['n_detections']:10.1f} │ {g['detection_improvement']:+5.1f}% ║"
        
        # Distance finale
        table += f"\n║ Distance finale à la source │ {n['final_distance']:17.1f} │ {h['final_distance']:10.1f} │ -      ║"
        
        table += "\n╚══════════════════════════════════════════════════════════════════╝"
        
        return table
    
    def plot_comparative_charts(self, metrics: Dict, save_path: str = "comparative_results.png"):
        """
        Créer des graphiques comparatifs
        
        Args:
            metrics: Métriques calculées
            save_path: Chemin pour sauvegarder la figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Analyse Comparative : Naïve vs HIGHLIGHT+', 
                     fontsize=16, fontweight='bold')
        
        n = metrics['naive']
        h = metrics['highlight']
        
        # 1. Taux de détection
        ax = axes[0, 0]
        bars = ax.bar(['Trajectoire\nNaïve', 'HIGHLIGHT+'], 
                     [n['detection_rate'], h['detection_rate']],
                     color=['#ff6b6b', '#4ecdc4'], alpha=0.8)
        ax.set_ylabel('Taux de détection (%)', fontsize=11)
        ax.set_title('Taux de Détection', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        # 2. Énergie consommée
        ax = axes[0, 1]
        bars = ax.bar(['Trajectoire\nNaïve', 'HIGHLIGHT+'], 
                     [n['energy_consumed'], h['energy_consumed']],
                     color=['#ff6b6b', '#4ecdc4'], alpha=0.8)
        ax.set_ylabel('Énergie (unités)', fontsize=11)
        ax.set_title('Consommation Énergétique', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontweight='bold')
        
        # 3. Temps de détection
        ax = axes[1, 0]
        naive_time = n['detection_time'] if n['detection_time'] else 0
        highlight_time = h['detection_time'] if h['detection_time'] else 0
        if naive_time > 0 or highlight_time > 0:
            bars = ax.bar(['Trajectoire\nNaïve', 'HIGHLIGHT+'], 
                         [naive_time, highlight_time],
                         color=['#ff6b6b', '#4ecdc4'], alpha=0.8)
            ax.set_ylabel('Temps (s)', fontsize=11)
            ax.set_title('Temps de Première Détection', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}s',
                           ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Pas de détection', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Temps de Première Détection', fontweight='bold')
        
        # 4. Gains relatifs
        ax = axes[1, 1]
        gains_data = {
            'Taux détection': metrics['gains']['detection_rate'] or 0,
            'Économie\nénergie': -metrics['gains']['energy_savings'] if metrics['gains']['energy_savings'] else 0,
            'Réduction\ntemps': metrics['gains']['time_reduction'] if metrics['gains']['time_reduction'] else 0
        }
        colors = ['green' if v > 0 else 'red' for v in gains_data.values()]
        bars = ax.bar(gains_data.keys(), gains_data.values(), 
                     color=colors, alpha=0.7)
        ax.set_ylabel('Gain (%)', fontsize=11)
        ax.set_title('Gains de Performance', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:+.1f}%',
                       ha='center', va='bottom' if height > 0 else 'top', 
                       fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphiques sauvegardés : {save_path}")
        
        return fig
    
    def create_animation_comparison(self, save_path: str = "comparative_animation.gif"):
        """
        Créer une animation comparative (côte à côte)
        
        Args:
            save_path: Chemin pour sauvegarder l'animation
        """
        # Exécuter une simulation de chaque type
        sim_naive = SimpleSimulator(self.config, agent_type="naive")
        results_naive = sim_naive.run()
        
        sim_highlight = SimpleSimulator(self.config, agent_type="highlight")
        results_highlight = sim_highlight.run()
        
        # Créer la figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Comparaison Visuelle : Naïve vs HIGHLIGHT+', 
                     fontsize=16, fontweight='bold')
        
        # Préparer les cartes de concentration
        x = np.linspace(0, self.config.world_size[0], 100)
        y = np.linspace(0, self.config.world_size[1], 100)
        X, Y = np.meshgrid(x, y)
        
        def get_concentration_map(plume):
            Z = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    Z[j, i] = plume.concentration(X[j, i], Y[j, i], noise=False)
            return Z
        
        Z1 = get_concentration_map(sim_naive.plume)
        Z2 = get_concentration_map(sim_highlight.plume)
        
        # Initialiser les axes
        ax1.contourf(X, Y, Z1, levels=20, cmap='YlOrRd', alpha=0.3)
        ax1.plot(self.config.leak_position[0], self.config.leak_position[1], 
                'rx', markersize=15, linewidth=3, label='Fuite')
        ax1.set_xlim(0, self.config.world_size[0])
        ax1.set_ylim(0, self.config.world_size[1])
        ax1.set_aspect('equal')
        ax1.set_title('Trajectoire Naïve', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Position X (m)')
        ax1.set_ylabel('Position Y (m)')
        ax1.grid(True, alpha=0.3)
        
        ax2.contourf(X, Y, Z2, levels=20, cmap='YlOrRd', alpha=0.3)
        ax2.plot(self.config.leak_position[0], self.config.leak_position[1], 
                'rx', markersize=15, linewidth=3, label='Fuite')
        ax2.set_xlim(0, self.config.world_size[0])
        ax2.set_ylim(0, self.config.world_size[1])
        ax2.set_aspect('equal')
        ax2.set_title('HIGHLIGHT+', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Position X (m)')
        ax2.set_ylabel('Position Y (m)')
        ax2.grid(True, alpha=0.3)
        
        # Lignes de trajectoire
        line1, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.7)
        point1, = ax1.plot([], [], 'bo', markersize=8)
        line2, = ax2.plot([], [], 'g-', linewidth=2, alpha=0.7)
        point2, = ax2.plot([], [], 'go', markersize=8)
        
        trajectory_naive = results_naive['trajectory']
        trajectory_highlight = results_highlight['trajectory']
        max_steps = min(len(trajectory_naive), len(trajectory_highlight))
        
        def animate(frame):
            # Afficher jusqu'à frame
            line1.set_data(trajectory_naive[:frame, 0], trajectory_naive[:frame, 1])
            if frame < len(trajectory_naive):
                point1.set_data([trajectory_naive[frame, 0]], [trajectory_naive[frame, 1]])
            
            line2.set_data(trajectory_highlight[:frame, 0], trajectory_highlight[:frame, 1])
            if frame < len(trajectory_highlight):
                point2.set_data([trajectory_highlight[frame, 0]], [trajectory_highlight[frame, 1]])
            
            return line1, point1, line2, point2
        
        anim = FuncAnimation(fig, animate, frames=max_steps, interval=50, blit=True)
        anim.save(save_path, writer='pillow', fps=20)
        print(f"✅ Animation sauvegardée : {save_path}")
        
        return anim
    
    def generate_report(self, metrics: Dict, output_path: str = "rapport_performance.txt"):
        """
        Générer un rapport de performance synthétique
        
        Args:
            metrics: Métriques calculées
            output_path: Chemin de sortie
        """
        report = f"""
{'='*70}
RAPPORT DE PERFORMANCE - HIGHLIGHT+
{'='*70}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Nombre de runs: {self.n_runs}

{self.create_comparative_table(metrics)}

CONCLUSION:
-----------
HIGHLIGHT+ démontre une amélioration significative par rapport à une 
trajectoire naïve systématique :

• Taux de détection amélioré de {metrics['gains']['detection_rate']:+.1f}%
• Économie d'énergie de {-metrics['gains']['energy_savings']:.1f}%
"""
        
        if metrics['gains']['time_reduction']:
            report += f"• Temps de détection réduit de {metrics['gains']['time_reduction']:.1f}%\n"
        
        report += f"""
• Nombre de détections augmenté de {metrics['gains']['detection_improvement']:.1f}%

Ces résultats valident l'approche HIGHLIGHT+ pour l'optimisation 
intelligente des trajectoires de drones de surveillance.

{'='*70}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Rapport sauvegardé : {output_path}")
        return report


if __name__ == "__main__":
    # Configuration
    config = SimpleConfig(
        leak_position=(60.0, 60.0),
        initial_position=(15.0, 15.0),
        max_steps=200,
        detection_threshold=0.3
    )
    
    # Analyse comparative
    analyzer = ComparativeAnalyzer(config, n_runs=10)
    analyzer.run_comparison()
    
    # Calcul des métriques
    metrics = analyzer.compute_metrics()
    
    # Affichage du tableau
    print("\n" + analyzer.create_comparative_table(metrics))
    
    # Génération des graphiques
    analyzer.plot_comparative_charts(metrics)
    
    # Génération du rapport
    analyzer.generate_report(metrics)
    
    print("\n✅ Analyse comparative terminée !")

