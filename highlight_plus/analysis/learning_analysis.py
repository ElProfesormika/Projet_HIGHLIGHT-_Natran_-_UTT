"""
Analyse de l'apprentissage pour HIGHLIGHT+
Détermine quand et comment le modèle apprend
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class LearningMetrics:
    """Métriques d'apprentissage"""
    step: int
    loss_rl: float
    loss_kl: float
    total_loss: float
    epsilon: float
    reward: float
    detection: bool
    concentration: float
    position: Tuple[float, float]


class LearningAnalyzer:
    """
    Analyseur de l'apprentissage pour HIGHLIGHT+
    
    Détermine :
    1. Quand le modèle apprend (convergence)
    2. Quand il peut détecter les fuites
    3. L'efficacité à travers les résultats
    4. Performance sur différentes positions de fuites
    """
    
    def __init__(self):
        self.learning_history = []
        self.detection_history = []
        self.performance_metrics = {}
        
    def add_learning_step(self, metrics: LearningMetrics):
        """Ajoute une étape d'apprentissage"""
        self.learning_history.append(metrics)
    
    def analyze_learning_convergence(self) -> Dict[str, Any]:
        """
        Analyse la convergence de l'apprentissage
        
        Returns:
            Dictionnaire avec les métriques de convergence
        """
        if not self.learning_history:
            return {}
        
        # Conversion en DataFrame pour analyse
        df = pd.DataFrame([
            {
                'step': m.step,
                'loss_rl': m.loss_rl,
                'loss_kl': m.loss_kl,
                'total_loss': m.total_loss,
                'epsilon': m.epsilon,
                'reward': m.reward,
                'detection': m.detection,
                'concentration': m.concentration
            }
            for m in self.learning_history
        ])
        
        # Analyse de convergence
        convergence_analysis = {}
        
        # 1. Convergence de la perte
        if len(df) > 100:
            # Moyenne mobile sur 50 étapes
            df['loss_ma'] = df['total_loss'].rolling(window=50).mean()
            df['loss_std'] = df['total_loss'].rolling(window=50).std()
            
            # Point de convergence (perte stable)
            stable_loss = df[df['loss_std'] < 0.01]
            if not stable_loss.empty:
                convergence_analysis['loss_convergence_step'] = stable_loss.iloc[0]['step']
                convergence_analysis['loss_convergence_value'] = stable_loss.iloc[0]['loss_ma']
        
        # 2. Convergence de l'exploration
        exploration_stable = df[df['epsilon'] < 0.1]
        if not exploration_stable.empty:
            convergence_analysis['exploration_convergence_step'] = exploration_stable.iloc[0]['step']
        
        # 3. Première détection
        first_detection = df[df['detection'] == True]
        if not first_detection.empty:
            convergence_analysis['first_detection_step'] = first_detection.iloc[0]['step']
            convergence_analysis['first_detection_concentration'] = first_detection.iloc[0]['concentration']
        
        # 4. Performance d'apprentissage
        convergence_analysis['learning_efficiency'] = self._calculate_learning_efficiency(df)
        
        return convergence_analysis
    
    def _calculate_learning_efficiency(self, df: pd.DataFrame) -> float:
        """Calcule l'efficacité d'apprentissage"""
        if len(df) < 100:
            return 0.0
        
        # Ratio détections / étapes
        detection_rate = df['detection'].sum() / len(df)
        
        # Amélioration de la récompense
        early_reward = df['reward'].iloc[:50].mean()
        late_reward = df['reward'].iloc[-50:].mean()
        reward_improvement = (late_reward - early_reward) / abs(early_reward) if early_reward != 0 else 0
        
        # Efficacité combinée
        efficiency = 0.6 * detection_rate + 0.4 * max(0, reward_improvement)
        
        return efficiency
    
    def analyze_detection_capability(self) -> Dict[str, Any]:
        """
        Analyse la capacité de détection du modèle
        
        Returns:
            Dictionnaire avec les métriques de détection
        """
        if not self.learning_history:
            return {}
        
        df = pd.DataFrame([
            {
                'step': m.step,
                'detection': m.detection,
                'concentration': m.concentration,
                'position': m.position
            }
            for m in self.learning_history
        ])
        
        detection_analysis = {}
        
        # 1. Taux de détection global
        detection_analysis['global_detection_rate'] = df['detection'].mean()
        
        # 2. Seuil de détection effectif
        detections = df[df['detection'] == True]
        if not detections.empty:
            detection_analysis['effective_detection_threshold'] = detections['concentration'].min()
            detection_analysis['mean_detection_concentration'] = detections['concentration'].mean()
        
        # 3. Évolution de la capacité de détection
        if len(df) > 100:
            # Détection par tranches de 100 étapes
            detection_by_epoch = []
            for i in range(0, len(df), 100):
                epoch_data = df.iloc[i:i+100]
                detection_rate = epoch_data['detection'].mean()
                detection_by_epoch.append(detection_rate)
            
            detection_analysis['detection_evolution'] = detection_by_epoch
            detection_analysis['detection_improvement'] = detection_by_epoch[-1] - detection_by_epoch[0] if len(detection_by_epoch) > 1 else 0
        
        # 4. Distance moyenne aux détections
        if not detections.empty:
            # Calcul de la distance moyenne aux positions de détection
            detection_positions = [m.position for m in self.learning_history if m.detection]
            if len(detection_positions) > 1:
                distances = []
                for i in range(1, len(detection_positions)):
                    dist = np.sqrt((detection_positions[i][0] - detection_positions[i-1][0])**2 + 
                                 (detection_positions[i][1] - detection_positions[i-1][1])**2)
                    distances.append(dist)
                detection_analysis['mean_detection_distance'] = np.mean(distances)
        
        return detection_analysis
    
    def plot_learning_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Trace les courbes d'apprentissage
        
        Args:
            save_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Figure matplotlib
        """
        if not self.learning_history:
            return None
        
        df = pd.DataFrame([
            {
                'step': m.step,
                'loss_rl': m.loss_rl,
                'loss_kl': m.loss_kl,
                'total_loss': m.total_loss,
                'epsilon': m.epsilon,
                'reward': m.reward,
                'detection': m.detection,
                'concentration': m.concentration
            }
            for m in self.learning_history
        ])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Évolution des pertes
        ax1 = axes[0, 0]
        ax1.plot(df['step'], df['loss_rl'], label='Perte RL', alpha=0.7)
        ax1.plot(df['step'], df['loss_kl'], label='Perte KL', alpha=0.7)
        ax1.plot(df['step'], df['total_loss'], label='Perte Totale', linewidth=2)
        ax1.set_xlabel('Étape')
        ax1.set_ylabel('Perte')
        ax1.set_title('Évolution des Pertes d\'Apprentissage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Exploration (epsilon)
        ax2 = axes[0, 1]
        ax2.plot(df['step'], df['epsilon'], 'g-', linewidth=2)
        ax2.set_xlabel('Étape')
        ax2.set_ylabel('Epsilon (Exploration)')
        ax2.set_title('Évolution de l\'Exploration')
        ax2.grid(True, alpha=0.3)
        
        # 3. Récompenses
        ax3 = axes[0, 2]
        # Moyenne mobile des récompenses
        df['reward_ma'] = df['reward'].rolling(window=50).mean()
        ax3.plot(df['step'], df['reward'], alpha=0.3, color='blue')
        ax3.plot(df['step'], df['reward_ma'], 'b-', linewidth=2, label='Moyenne mobile')
        ax3.set_xlabel('Étape')
        ax3.set_ylabel('Récompense')
        ax3.set_title('Évolution des Récompenses')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Détections
        ax4 = axes[1, 0]
        detections = df[df['detection'] == True]
        if not detections.empty:
            ax4.scatter(detections['step'], detections['concentration'], 
                       c='red', s=50, alpha=0.7, label='Détections')
        ax4.plot(df['step'], df['concentration'], 'b-', alpha=0.5, label='Concentration')
        ax4.set_xlabel('Étape')
        ax4.set_ylabel('Concentration (kg/m³)')
        ax4.set_title('Détections vs Concentration')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Taux de détection par époque
        ax5 = axes[1, 1]
        if len(df) > 100:
            detection_by_epoch = []
            steps_by_epoch = []
            for i in range(0, len(df), 100):
                epoch_data = df.iloc[i:i+100]
                detection_rate = epoch_data['detection'].mean()
                detection_by_epoch.append(detection_rate)
                steps_by_epoch.append(epoch_data['step'].iloc[-1])
            
            ax5.plot(steps_by_epoch, detection_by_epoch, 'ro-', linewidth=2, markersize=6)
            ax5.set_xlabel('Étape')
            ax5.set_ylabel('Taux de Détection')
            ax5.set_title('Évolution du Taux de Détection')
            ax5.grid(True, alpha=0.3)
        
        # 6. Métriques de convergence
        ax6 = axes[1, 2]
        convergence_analysis = self.analyze_learning_convergence()
        detection_analysis = self.analyze_detection_capability()
        
        metrics_text = f"""
        Convergence de la perte: {convergence_analysis.get('loss_convergence_step', 'N/A')}
        Première détection: {convergence_analysis.get('first_detection_step', 'N/A')}
        Taux de détection: {detection_analysis.get('global_detection_rate', 0):.3f}
        Efficacité apprentissage: {convergence_analysis.get('learning_efficiency', 0):.3f}
        Seuil effectif: {detection_analysis.get('effective_detection_threshold', 'N/A'):.4f}
        """
        
        ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax6.set_title('Métriques de Convergence')
        ax6.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_learning_analyzer() -> LearningAnalyzer:
    """Crée un analyseur d'apprentissage"""
    return LearningAnalyzer()


if __name__ == "__main__":
    # Test de l'analyseur
    analyzer = create_learning_analyzer()
    
    # Simulation de données d'apprentissage
    np.random.seed(42)
    
    for step in range(1000):
        # Simulation de l'apprentissage
        loss_rl = 1.0 * np.exp(-step/200) + 0.1 * np.random.normal()
        loss_kl = 0.5 * np.exp(-step/300) + 0.05 * np.random.normal()
        total_loss = loss_rl + 0.1 * loss_kl
        epsilon = max(0.01, 1.0 - step/500)
        reward = 2.0 * (1 - np.exp(-step/100)) + 0.5 * np.random.normal()
        
        # Simulation de détections (plus fréquentes avec le temps)
        detection_prob = min(0.8, step/800)
        detection = np.random.random() < detection_prob
        concentration = np.random.exponential(0.1) if detection else np.random.exponential(0.01)
        
        position = (np.random.uniform(0, 100), np.random.uniform(0, 100))
        
        metrics = LearningMetrics(
            step=step,
            loss_rl=loss_rl,
            loss_kl=loss_kl,
            total_loss=total_loss,
            epsilon=epsilon,
            reward=reward,
            detection=detection,
            concentration=concentration,
            position=position
        )
        
        analyzer.add_learning_step(metrics)
    
    # Analyse
    convergence = analyzer.analyze_learning_convergence()
    detection = analyzer.analyze_detection_capability()
    
    print("Analyse de l'apprentissage:")
    print("=" * 40)
    print(f"Convergence de la perte: étape {convergence.get('loss_convergence_step', 'N/A')}")
    print(f"Première détection: étape {convergence.get('first_detection_step', 'N/A')}")
    print(f"Taux de détection global: {detection.get('global_detection_rate', 0):.3f}")
    print(f"Efficacité d'apprentissage: {convergence.get('learning_efficiency', 0):.3f}")
    
    # Visualisation
    fig = analyzer.plot_learning_curves()
    plt.show()
