"""
Simulateur de capteur TDLAS (Tunable Diode Laser Absorption Spectroscopy)
pour la détection de méthane dans le cadre du projet HIGHLIGHT+
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats


@dataclass
class TDLASConfig:
    """Configuration du capteur TDLAS"""
    # Caractéristiques du capteur
    noise_level: float = 0.1          # Niveau de bruit (σ)
    detection_threshold: float = 0.05  # Seuil de détection (kg/m³)
    range_max: float = 100.0          # Portée maximale (m)
    range_min: float = 1.0            # Portée minimale (m)
    
    # Paramètres de mesure
    update_frequency: float = 10.0    # Fréquence de mise à jour (Hz)
    measurement_time: float = 0.1     # Temps d'intégration (s)
    
    # Caractéristiques physiques
    laser_wavelength: float = 1653.7  # Longueur d'onde (nm) - CH4
    beam_divergence: float = 0.1      # Divergence du faisceau (mrad)
    power: float = 10.0               # Puissance laser (mW)
    
    # Bruit et interférences
    atmospheric_noise: float = 0.05   # Bruit atmosphérique
    electronic_noise: float = 0.02    # Bruit électronique
    interference_factor: float = 0.1  # Facteur d'interférence


class TDLASSensor:
    """
    Simulateur de capteur TDLAS pour la détection de méthane
    
    Le capteur TDLAS fonctionne selon le principe de l'absorption différentielle :
    - Un laser émet à une longueur d'onde spécifique au méthane
    - Le signal est réfléchi par une surface et détecté
    - L'absorption est proportionnelle à la concentration de méthane
    
    Modèle mathématique :
    I_detected = I_0 * exp(-α * C * L) + noise
    
    où :
    - I_0 : intensité laser initiale
    - α : coefficient d'absorption du méthane
    - C : concentration de méthane
    - L : longueur du trajet optique
    - noise : bruit du système
    """
    
    def __init__(self, config: TDLASConfig):
        self.config = config
        self._absorption_coefficient = self._compute_absorption_coefficient()
        self._measurement_history = []
        
    def _compute_absorption_coefficient(self) -> float:
        """
        Calcule le coefficient d'absorption du méthane à la longueur d'onde du laser
        
        Pour CH4 à 1653.7 nm, le coefficient d'absorption est d'environ 0.1 m²/kg
        """
        # Coefficient d'absorption typique pour CH4 à 1653.7 nm
        return 0.1  # m²/kg
    
    def measure_concentration(self, true_concentration: float, 
                            distance: float,
                            surface_reflectivity: float = 0.3,
                            temperature: float = 288.15,
                            pressure: float = 101325) -> Tuple[float, bool]:
        """
        Simule une mesure de concentration de méthane
        
        Implémente explicitement la formule : I_s ∝ ρ / h²
        
        Args:
            true_concentration: Concentration réelle (kg/m³)
            distance: Distance au capteur = altitude h (m)
            surface_reflectivity: Réflectivité de la surface ρ (0-1)
            temperature: Température (K)
            pressure: Pression (Pa)
            
        Returns:
            Tuple (measured_concentration, detection_flag)
        """
        # Vérification de la portée
        if distance < self.config.range_min or distance > self.config.range_max:
            return 0.0, False
        
        # Calcul de l'intensité détectée selon la loi de Beer-Lambert
        # avec dépendance explicite en altitude : I_s ∝ ρ / h²
        # I = I_0 * exp(-α * C * L) * (ρ / h²)
        # où L est la longueur du trajet optique (2 * distance pour aller-retour)
        path_length = 2 * distance
        
        # Intensité théorique selon Beer-Lambert
        intensity_ratio = np.exp(-self._absorption_coefficient * true_concentration * path_length)
        
        # Facteur géométrique avec dépendance explicite en altitude
        # Selon le document : I_s ∝ ρ / h²
        altitude = distance  # Pour un capteur pointant vers le sol, distance = altitude
        geometric_factor = surface_reflectivity / (altitude ** 2)
        
        # Intensité détectée (normalisée par I_0)
        detected_intensity = intensity_ratio * geometric_factor
        
        # Conversion en concentration mesurée
        if detected_intensity > 0:
            measured_conc = -np.log(detected_intensity / geometric_factor) / \
                           (self._absorption_coefficient * path_length)
        else:
            measured_conc = 0.0
        
        # Ajout du bruit
        noise = self._generate_noise(distance, temperature, pressure)
        measured_conc += noise
        
        # Seuil de détection
        detection = measured_conc > self.config.detection_threshold
        
        # Stockage de l'historique
        self._measurement_history.append({
            'true_concentration': true_concentration,
            'measured_concentration': measured_conc,
            'distance': distance,
            'detection': detection,
            'noise': noise
        })
        
        return max(0.0, measured_conc), detection
    
    def _generate_noise(self, distance: float, temperature: float, 
                       pressure: float) -> float:
        """
        Génère le bruit du capteur selon différents composants
        
        Le bruit total est la somme de :
        - Bruit électronique (constant)
        - Bruit atmosphérique (dépend de la distance)
        - Bruit d'interférence (aléatoire)
        """
        # Bruit électronique (constant)
        electronic_noise = np.random.normal(0, self.config.electronic_noise)
        
        # Bruit atmosphérique (augmente avec la distance)
        atmospheric_noise = np.random.normal(0, 
            self.config.atmospheric_noise * np.sqrt(distance / 10.0))
        
        # Bruit d'interférence (aléatoire)
        interference_noise = np.random.normal(0, 
            self.config.interference_factor * self.config.noise_level)
        
        # Bruit total
        total_noise = electronic_noise + atmospheric_noise + interference_noise
        
        return total_noise
    
    def measure_at_position(self, x: float, y: float, z: float,
                          plume_concentration: float,
                          surface_reflectivity: float = 0.3) -> Tuple[float, bool]:
        """
        Mesure la concentration à une position donnée
        
        Args:
            x, y, z: Position du capteur (m)
            plume_concentration: Concentration du panache à cette position
            surface_reflectivity: Réflectivité de la surface
            
        Returns:
            Tuple (measured_concentration, detection_flag)
        """
        # Distance au sol (hypothèse : mesure vers le sol)
        distance = z
        
        return self.measure_concentration(
            plume_concentration, distance, surface_reflectivity
        )
    
    def get_signal_to_noise_ratio(self, concentration: float, 
                                 distance: float,
                                 surface_reflectivity: float = 0.3) -> float:
        """
        Calcule le rapport signal/bruit (SNR) pour une concentration donnée
        
        Intègre explicitement la dépendance en altitude : I_s ∝ ρ / h²
        
        Args:
            concentration: Concentration de méthane (kg/m³)
            distance: Distance de mesure = altitude h (m)
            surface_reflectivity: Réflectivité de la surface ρ (0-1)
            
        Returns:
            SNR (dimensionless)
        """
        # Signal théorique avec dépendance en altitude
        # I_s = I_0 * exp(-α * C * L) * (ρ / h²)
        path_length = 2 * distance
        altitude = distance
        
        # Signal absorbé (Beer-Lambert)
        absorption_signal = concentration * self._absorption_coefficient * path_length
        
        # Signal géométrique (dépendance altitude selon document)
        # I_s ∝ ρ / h²
        geometric_signal = surface_reflectivity / (altitude ** 2)
        
        # Signal total (produit des deux contributions)
        signal = absorption_signal * geometric_signal
        
        # Bruit total (augmente avec la distance/altitude)
        noise_std = np.sqrt(
            self.config.electronic_noise**2 + 
            (self.config.atmospheric_noise * np.sqrt(distance / 10.0))**2 +
            (self.config.interference_factor * self.config.noise_level)**2 +
            # Bruit supplémentaire dû à l'altitude (dégradation du signal)
            (0.01 * distance / 10.0)**2  # Bruit proportionnel à l'altitude
        )
        
        return signal / noise_std if noise_std > 0 else 0.0
    
    def plot_measurement_history(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Visualise l'historique des mesures
        
        Args:
            ax: Axes matplotlib existants (optionnel)
            
        Returns:
            Axes matplotlib avec la visualisation
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        if not self._measurement_history:
            ax.text(0.5, 0.5, 'Aucune mesure disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            return ax
        
        # Extraction des données
        times = np.arange(len(self._measurement_history))
        true_conc = [m['true_concentration'] for m in self._measurement_history]
        measured_conc = [m['measured_concentration'] for m in self._measurement_history]
        detections = [m['detection'] for m in self._measurement_history]
        
        # Tracé des concentrations
        ax.plot(times, true_conc, 'b-', label='Concentration réelle', linewidth=2)
        ax.plot(times, measured_conc, 'r--', label='Concentration mesurée', linewidth=1)
        
        # Marquage des détections
        detection_times = [t for t, d in enumerate(detections) if d]
        detection_conc = [measured_conc[t] for t in detection_times]
        ax.scatter(detection_times, detection_conc, c='green', s=50, 
                  label='Détections', zorder=5)
        
        # Seuil de détection
        ax.axhline(y=self.config.detection_threshold, color='orange', 
                  linestyle=':', label='Seuil de détection')
        
        # Configuration
        ax.set_xlabel('Numéro de mesure')
        ax.set_ylabel('Concentration (kg/m³)')
        ax.set_title('Historique des mesures TDLAS')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def get_statistics(self) -> dict:
        """
        Retourne les statistiques des mesures
        
        Returns:
            Dictionnaire avec les statistiques
        """
        if not self._measurement_history:
            return {}
        
        true_conc = [m['true_concentration'] for m in self._measurement_history]
        measured_conc = [m['measured_concentration'] for m in self._measurement_history]
        detections = [m['detection'] for m in self._measurement_history]
        
        # Calcul des métriques
        detection_rate = np.mean(detections)
        mse = np.mean((np.array(true_conc) - np.array(measured_conc))**2)
        mae = np.mean(np.abs(np.array(true_conc) - np.array(measured_conc)))
        
        # SNR moyen
        snr_values = []
        for m in self._measurement_history:
            snr = self.get_signal_to_noise_ratio(m['true_concentration'], m['distance'])
            snr_values.append(snr)
        
        return {
            'total_measurements': len(self._measurement_history),
            'detection_rate': detection_rate,
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'mean_snr': np.mean(snr_values),
            'std_snr': np.std(snr_values)
        }
    
    def reset(self):
        """Remet à zéro l'historique des mesures"""
        self._measurement_history = []


def create_test_sensor() -> TDLASSensor:
    """Crée un capteur TDLAS de test avec des paramètres réalistes"""
    config = TDLASConfig(
        noise_level=0.08,
        detection_threshold=0.03,
        range_max=50.0,
        range_min=1.0,
        update_frequency=20.0
    )
    return TDLASSensor(config)


if __name__ == "__main__":
    # Test du capteur
    sensor = create_test_sensor()
    
    # Test avec différentes concentrations
    concentrations = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
    distances = [5.0, 10.0, 15.0, 20.0]
    
    print("Test du capteur TDLAS:")
    print("=" * 50)
    
    for conc in concentrations:
        for dist in distances:
            measured, detected = sensor.measure_concentration(conc, dist)
            snr = sensor.get_signal_to_noise_ratio(conc, dist)
            print(f"Conc: {conc:.3f}, Dist: {dist:.1f}m -> "
                  f"Mesuré: {measured:.3f}, Détecté: {detected}, SNR: {snr:.2f}")
    
    # Visualisation de l'historique
    plt.figure(figsize=(12, 6))
    sensor.plot_measurement_history()
    plt.show()
    
    # Statistiques
    stats = sensor.get_statistics()
    print("\nStatistiques:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")




