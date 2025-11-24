"""
Apprenti (Student) utilisant l'apprentissage par renforcement pour HIGHLIGHT+
Implémentation avec distillation de connaissance depuis l'Expert (Teacher)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import deque
import random

# Import des composants du projet
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.teacher_gp import GaussianProcessTeacher, TeacherConfig


@dataclass
class StudentConfig:
    """Configuration de l'Apprenti (Student)"""
    # Architecture du réseau
    hidden_layers: List[int] = None
    activation: str = "tanh"
    learning_rate: float = 3e-4
    
    # Hyperparamètres d'entraînement
    batch_size: int = 64
    buffer_size: int = 10000
    target_update_freq: int = 100
    learning_starts: int = 1000
    
    # Distillation de connaissance
    lambda_kl: float = 0.1      # Poids de la perte KL
    temperature: float = 3.0    # Température de distillation
    teacher_update_freq: int = 10  # Fréquence de mise à jour du teacher
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    
    # Récompense
    gamma: float = 0.99         # Facteur de discount
    reward_scale: float = 1.0   # Échelle des récompenses


class ReplayBuffer:
    """Buffer d'expérience pour l'apprentissage par renforcement"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """Ajoute une expérience au buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Échantillonne un batch d'expériences"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (torch.FloatTensor(state),
                torch.FloatTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.BoolTensor(done))
    
    def __len__(self) -> int:
        return len(self.buffer)


class StudentNetwork(nn.Module):
    """Réseau de neurones pour l'Apprenti (Student)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int], 
                 activation: str = "tanh"):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Construction des couches
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Actions normalisées [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Passe avant du réseau"""
        return self.network(state)


class StudentRL:
    """
    Apprenti (Student) utilisant l'apprentissage par renforcement
    
    L'Apprenti apprend une politique de navigation en combinant :
    1. L'apprentissage par renforcement classique (récompenses environnementales)
    2. La distillation de connaissance depuis l'Expert (Teacher)
    
    Architecture :
    - Réseau de neurones pour la politique π_θ(s) → a
    - Buffer d'expérience pour l'apprentissage hors-ligne
    - Perte combinée : L = L_RL + λ * L_KL(π_teacher || π_student)
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: StudentConfig,
                 teacher: Optional[GaussianProcessTeacher] = None):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.teacher = teacher
        
        # Réseau principal
        if config.hidden_layers is None:
            config.hidden_layers = [256, 256, 128]
        
        self.policy_net = StudentNetwork(state_dim, action_dim, 
                                       config.hidden_layers, config.activation)
        self.target_net = StudentNetwork(state_dim, action_dim, 
                                       config.hidden_layers, config.activation)
        
        # Optimiseur
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                  lr=config.learning_rate)
        
        # Buffer d'expérience
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Paramètres d'exploration (améliorés)
        self.epsilon = config.epsilon_start
        self.epsilon_decay = (config.epsilon_start - config.epsilon_end) / config.epsilon_decay
        self.epsilon_min = config.epsilon_end  # Minimum d'exploration
        
        # Compteurs
        self.step_count = 0
        self.episode_count = 0
        
        # Historique
        self.loss_history = []
        self.reward_history = []
        self.kl_loss_history = []
        
        # Initialisation du réseau cible
        self.update_target_network()
    
    def select_action(self, state: np.ndarray, training: bool = True, 
                     teacher_guidance: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sélectionne une action selon la politique actuelle
        
        Args:
            state: État actuel
            training: Mode d'entraînement (exploration vs exploitation)
            teacher_guidance: Action suggérée par le Teacher (optionnel, pour guidance)
            
        Returns:
            Action sélectionnée
        """
        if training and random.random() < self.epsilon:
            # Exploration guidée par le Teacher si disponible
            if teacher_guidance is not None and self.teacher is not None:
                # Exploration autour de la direction du Teacher (plus intelligente)
                # S'assurer que teacher_guidance a la bonne shape (3,)
                if len(teacher_guidance) == 2:
                    teacher_guidance_3d = np.append(teacher_guidance, 0.0)  # Ajouter z=0
                else:
                    teacher_guidance_3d = teacher_guidance[:self.action_dim]
                noise = np.random.uniform(-0.3, 0.3, self.action_dim)
                action = teacher_guidance_3d + noise
                action = np.clip(action, -1, 1)
            else:
                # Exploration aléatoire guidée (moins aléatoire, plus directionnelle)
                action = np.random.uniform(-0.5, 0.5, self.action_dim)  # Réduire l'amplitude
        else:
            # Exploitation de la politique
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_tensor = self.policy_net(state_tensor)
                action = action_tensor.squeeze(0).numpy()
                
                # Si le Student n'est pas encore bien entraîné, mélanger avec guidance Teacher
                if teacher_guidance is not None and len(self.loss_history) < 50:
                    # Au début, favoriser légèrement le Teacher
                    # S'assurer que teacher_guidance a la bonne shape (3,)
                    if len(teacher_guidance) == 2:
                        teacher_guidance_3d = np.append(teacher_guidance, 0.0)  # Ajouter z=0
                    else:
                        teacher_guidance_3d = teacher_guidance[:self.action_dim]
                    action = 0.7 * action + 0.3 * teacher_guidance_3d
                    action = np.clip(action, -1, 1)
        
        return action.astype(np.float32)
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Stocke une expérience dans le buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def learn(self) -> Dict[str, float]:
        """
        Effectue une étape d'apprentissage
        
        Returns:
            Dictionnaire avec les métriques d'apprentissage
        """
        if len(self.replay_buffer) < self.config.learning_starts:
            return {}
        
        # Échantillonnage du batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        # Calcul de la perte RL
        rl_loss = self._compute_rl_loss(states, actions, rewards, next_states, dones)
        
        # Calcul de la perte de distillation
        kl_loss = torch.tensor(0.0)
        if self.teacher is not None and self.step_count % self.config.teacher_update_freq == 0:
            kl_loss = self._compute_kl_loss(states)
        
        # Perte totale
        total_loss = rl_loss + self.config.lambda_kl * kl_loss
        
        # Mise à jour des paramètres
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Mise à jour du réseau cible
        if self.step_count % self.config.target_update_freq == 0:
            self.update_target_network()
        
        # Mise à jour de l'exploration (décroissance adaptative)
        # Plus lent si on apprend bien (perte faible)
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            avg_loss = np.mean(recent_losses)
            # Si perte faible, réduire l'exploration plus vite
            if avg_loss < 0.1:
                self.epsilon = max(self.config.epsilon_end, 
                                 self.epsilon - self.epsilon_decay * 1.5)
            else:
                self.epsilon = max(self.config.epsilon_end, 
                                 self.epsilon - self.epsilon_decay)
        else:
            self.epsilon = max(self.config.epsilon_end, 
                             self.epsilon - self.epsilon_decay)
        
        # Enregistrement des métriques
        metrics = {
            'rl_loss': rl_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item(),
            'epsilon': self.epsilon
        }
        
        self.loss_history.append(total_loss.item())
        self.kl_loss_history.append(kl_loss.item())
        
        return metrics
    
    def _compute_rl_loss(self, states: torch.Tensor, actions: torch.Tensor,
                        rewards: torch.Tensor, next_states: torch.Tensor,
                        dones: torch.Tensor) -> torch.Tensor:
        """Calcule la perte d'apprentissage par renforcement"""
        # Prédictions actuelles
        current_q = self.policy_net(states)
        
        # Prédictions cibles
        with torch.no_grad():
            next_q = self.target_net(next_states)
            target_q = rewards + self.config.gamma * next_q * (~dones)
        
        # Perte MSE
        loss = F.mse_loss(current_q, target_q)
        return loss
    
    def _compute_kl_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calcule la perte de distillation de connaissance
        
        L_KL = D_KL(π_teacher || π_student)
        """
        if self.teacher is None:
            return torch.tensor(0.0)
        
        # Prédictions du student
        student_actions = self.policy_net(states)
        
        # Prédictions du teacher (simulées)
        teacher_actions = self._get_teacher_actions(states)
        
        # Calcul de la divergence KL
        # Approximation : MSE entre les actions
        kl_loss = F.mse_loss(student_actions, teacher_actions)
        
        return kl_loss
    
    def _get_teacher_actions(self, states: torch.Tensor) -> torch.Tensor:
        """
        Obtient les actions recommandées par le Teacher
        
        Note: Cette méthode simule les recommandations du Teacher.
        Dans une implémentation complète, elle ferait appel au Teacher
        pour obtenir les actions optimales.
        """
        # Simulation des actions du teacher basées sur l'état
        batch_size = states.shape[0]
        
        # Actions simulées (dans un vrai système, ceci viendrait du Teacher)
        teacher_actions = torch.zeros_like(states[:, :self.action_dim])
        
        # Simulation basée sur la position et le gradient
        for i in range(batch_size):
            state = states[i].numpy()
            
            # Position actuelle
            pos_x = state[0] * 100  # Dénormalisation
            pos_y = state[1] * 100
            
            # Gradient (simulé)
            grad_x = state[8]
            grad_y = state[9]
            
            # Action basée sur le gradient (direction vers la source)
            action_x = np.clip(grad_x, -1, 1)
            action_y = np.clip(grad_y, -1, 1)
            action_z = 0.0  # Pas de mouvement vertical pour simplifier
            
            teacher_actions[i] = torch.tensor([action_x, action_y, action_z])
        
        return teacher_actions
    
    def update_target_network(self):
        """Met à jour le réseau cible"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filepath: str):
        """Sauvegarde le modèle"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        """Charge un modèle sauvegardé"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.epsilon = checkpoint['epsilon']
    
    def plot_training_progress(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Visualise les progrès d'entraînement"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Tracé des pertes
        if self.loss_history:
            ax.plot(self.loss_history, label='Perte totale', alpha=0.7)
            ax.plot(self.kl_loss_history, label='Perte KL', alpha=0.7)
        
        ax.set_xlabel('Étapes d\'entraînement')
        ax.set_ylabel('Perte')
        ax.set_title('Progrès d\'entraînement de l\'Apprenti')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de performance"""
        return {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'mean_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0.0,
            'mean_kl_loss': np.mean(self.kl_loss_history[-100:]) if self.kl_loss_history else 0.0
        }
    
    def reset(self):
        """Remet à zéro l'Apprenti"""
        self.step_count = 0
        self.episode_count = 0
        self.epsilon = self.config.epsilon_start
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        self.loss_history = []
        self.reward_history = []
        self.kl_loss_history = []


def create_test_student(teacher: Optional[GaussianProcessTeacher] = None) -> StudentRL:
    """Crée un Apprenti de test avec des paramètres réalistes"""
    config = StudentConfig(
        hidden_layers=[128, 64],
        learning_rate=1e-3,
        batch_size=32,
        buffer_size=5000,
        lambda_kl=0.2,
        temperature=2.0,
        epsilon_decay=5000
    )
    
    state_dim = 16  # Dimension de l'observation complète (selon MDP du document)
    action_dim = 3  # Dimension de l'action
    
    return StudentRL(state_dim, action_dim, config, teacher)


if __name__ == "__main__":
    # Test de l'Apprenti
    print("Test de l'Apprenti (Student):")
    print("=" * 40)
    
    # Création d'un Teacher de test
    from models.teacher_gp import create_test_teacher
    teacher = create_test_teacher()
    
    # Création de l'Apprenti
    student = create_test_student(teacher)
    
    # Test de sélection d'action
    state = np.random.randn(11)
    action = student.select_action(state)
    print(f"État: {state}")
    print(f"Action sélectionnée: {action}")
    
    # Test d'apprentissage
    for i in range(100):
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
    print(f"Métriques d'apprentissage: {metrics}")
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    student.plot_training_progress()
    plt.show()
    
    # Métriques de performance
    perf_metrics = student.get_performance_metrics()
    print(f"Métriques de performance: {perf_metrics}")




