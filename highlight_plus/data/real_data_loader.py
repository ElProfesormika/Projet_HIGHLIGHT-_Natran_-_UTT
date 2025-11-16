"""
Chargeur de données réelles pour HIGHLIGHT+
Intégration de données de terrain pour l'entraînement et la validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class RealDataConfig:
    """Configuration pour le chargement de données réelles"""
    data_path: str = "data/real_measurements/"
    file_format: str = "csv"  # csv, json, hdf5
    coordinate_system: str = "UTM"  # UTM, WGS84, local
    concentration_unit: str = "ppm"  # ppm, ppb, kg/m³
    time_column: str = "timestamp"
    x_column: str = "x"
    y_column: str = "y"
    concentration_column: str = "ch4_concentration"
    quality_threshold: float = 0.8  # Seuil de qualité des données


class RealDataLoader:
    """
    Chargeur de données réelles de mesure de méthane
    
    Supporte différents formats de données :
    - Données de capteurs fixes
    - Données de drones de surveillance
    - Données de campagnes de mesure
    - Données satellitaires (TROPOMI, GOSAT)
    """
    
    def __init__(self, config: RealDataConfig):
        self.config = config
        self.data_path = Path(config.data_path)
        self.measurements = []
        self.metadata = {}
        
    def load_measurements(self, file_path: str) -> pd.DataFrame:
        """
        Charge les mesures depuis un fichier
        
        Args:
            file_path: Chemin vers le fichier de données
            
        Returns:
            DataFrame avec les mesures
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
        
        # Chargement selon le format
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        elif file_path.suffix.lower() in ['.h5', '.hdf5']:
            df = pd.read_hdf(file_path)
        else:
            raise ValueError(f"Format non supporté: {file_path.suffix}")
        
        # Validation des colonnes
        required_columns = [self.config.x_column, self.config.y_column, 
                          self.config.concentration_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Colonnes manquantes: {missing_columns}")
        
        # Nettoyage des données
        df = self._clean_data(df)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie et valide les données"""
        # Suppression des valeurs manquantes
        df = df.dropna(subset=[self.config.x_column, self.config.y_column, 
                              self.config.concentration_column])
        
        # Filtrage des valeurs aberrantes
        conc_col = self.config.concentration_column
        q1 = df[conc_col].quantile(0.25)
        q3 = df[conc_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        df = df[(df[conc_col] >= lower_bound) & (df[conc_col] <= upper_bound)]
        
        # Conversion des unités si nécessaire
        if self.config.concentration_unit == "ppm":
            # Conversion ppm vers kg/m³ (approximative)
            df[conc_col] = df[conc_col] * 0.000717  # kg/m³ à 20°C, 1 atm
        elif self.config.concentration_unit == "ppb":
            df[conc_col] = df[conc_col] * 0.000000717
        
        return df
    
    def create_training_dataset(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Crée un dataset d'entraînement à partir des données réelles
        
        Args:
            df: DataFrame avec les mesures
            
        Returns:
            Dictionnaire avec les données formatées
        """
        # Extraction des coordonnées et concentrations
        x = df[self.config.x_column].values
        y = df[self.config.y_column].values
        concentrations = df[self.config.concentration_column].values
        
        # Normalisation des coordonnées
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        
        # Création du dataset
        dataset = {
            'positions': np.column_stack([x_norm, y_norm]),
            'concentrations': concentrations,
            'raw_positions': np.column_stack([x, y]),
            'metadata': {
                'n_samples': len(df),
                'x_range': (x.min(), x.max()),
                'y_range': (y.min(), y.max()),
                'concentration_range': (concentrations.min(), concentrations.max()),
                'source_file': str(df.attrs.get('source_file', 'unknown'))
            }
        }
        
        return dataset
    
    def simulate_plume_from_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Simule un panache à partir des données réelles
        
        Args:
            df: DataFrame avec les mesures
            
        Returns:
            Dictionnaire avec les paramètres du panache simulé
        """
        # Estimation des paramètres du panache
        concentrations = df[self.config.concentration_column].values
        positions = df[[self.config.x_column, self.config.y_column]].values
        
        # Estimation de la position de la source (point de concentration maximale)
        max_idx = np.argmax(concentrations)
        source_position = positions[max_idx]
        
        # Estimation de l'intensité de la source
        source_intensity = concentrations[max_idx]
        
        # Estimation des paramètres de diffusion
        # Calcul de la variance dans les directions x et y
        centered_positions = positions - source_position
        sigma_x = np.std(centered_positions[:, 0])
        sigma_y = np.std(centered_positions[:, 1])
        
        # Estimation de la direction du vent (basée sur l'asymétrie)
        # Simplification : direction principale de dispersion
        if sigma_x > sigma_y:
            wind_direction = 0  # Vent principalement en X
        else:
            wind_direction = 90  # Vent principalement en Y
        
        plume_params = {
            'leak_x': source_position[0],
            'leak_y': source_position[1],
            'leak_intensity': source_intensity,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'wind_direction': wind_direction,
            'wind_speed': 2.0,  # Valeur par défaut
            'data_quality': len(df) / 1000.0  # Score de qualité basé sur le nombre de points
        }
        
        return plume_params
    
    def plot_real_data(self, df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualise les données réelles
        
        Args:
            df: DataFrame avec les mesures
            save_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Figure matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Carte de concentration
        x = df[self.config.x_column]
        y = df[self.config.y_column]
        concentrations = df[self.config.concentration_column]
        
        scatter = ax1.scatter(x, y, c=concentrations, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Position X (m)')
        ax1.set_ylabel('Position Y (m)')
        ax1.set_title('Données Réelles - Carte de Concentration')
        plt.colorbar(scatter, ax=ax1, label='Concentration (kg/m³)')
        
        # Distribution des concentrations
        ax2.hist(concentrations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Concentration (kg/m³)')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution des Concentrations')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_for_simulation(self, df: pd.DataFrame, output_path: str):
        """
        Exporte les données dans un format compatible avec la simulation
        
        Args:
            df: DataFrame avec les mesures
            output_path: Chemin de sortie
        """
        # Création du dataset d'entraînement
        dataset = self.create_training_dataset(df)
        
        # Paramètres du panache simulé
        plume_params = self.simulate_plume_from_data(df)
        
        # Export
        export_data = {
            'dataset': dataset,
            'plume_parameters': plume_params,
            'config': {
                'coordinate_system': self.config.coordinate_system,
                'concentration_unit': self.config.concentration_unit,
                'data_quality': plume_params['data_quality']
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Données exportées vers: {output_path}")


def create_sample_real_data() -> pd.DataFrame:
    """
    Crée des données d'exemple basées sur des patterns réels
    
    Returns:
        DataFrame avec des données simulées réalistes
    """
    np.random.seed(42)
    
    # Simulation d'une fuite de méthane
    n_points = 500
    
    # Position de la source
    source_x, source_y = 50.0, 50.0
    
    # Génération de points autour de la source
    angles = np.random.uniform(0, 2*np.pi, n_points)
    distances = np.random.exponential(10, n_points)
    
    x = source_x + distances * np.cos(angles)
    y = source_y + distances * np.sin(angles)
    
    # Concentration basée sur la distance (loi en 1/r²)
    distances_from_source = np.sqrt((x - source_x)**2 + (y - source_y)**2)
    base_concentration = 0.5 / (1 + distances_from_source/10)
    
    # Ajout de bruit réaliste
    noise = np.random.normal(0, 0.05, n_points)
    concentrations = np.maximum(0, base_concentration + noise)
    
    # Création du DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'ch4_concentration': concentrations,
        'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='1min')
    })
    
    return df


if __name__ == "__main__":
    # Test du chargeur de données
    config = RealDataConfig()
    loader = RealDataLoader(config)
    
    # Création de données d'exemple
    sample_data = create_sample_real_data()
    
    print("Test du chargeur de données réelles:")
    print("=" * 40)
    print(f"Nombre de points: {len(sample_data)}")
    print(f"Concentration moyenne: {sample_data['ch4_concentration'].mean():.4f} kg/m³")
    print(f"Concentration max: {sample_data['ch4_concentration'].max():.4f} kg/m³")
    
    # Création du dataset d'entraînement
    dataset = loader.create_training_dataset(sample_data)
    print(f"\nDataset créé avec {dataset['metadata']['n_samples']} échantillons")
    
    # Paramètres du panache
    plume_params = loader.simulate_plume_from_data(sample_data)
    print(f"\nParamètres du panache estimés:")
    print(f"  - Source: ({plume_params['leak_x']:.1f}, {plume_params['leak_y']:.1f})")
    print(f"  - Intensité: {plume_params['leak_intensity']:.4f} kg/m³")
    print(f"  - Diffusion: σx={plume_params['sigma_x']:.1f}, σy={plume_params['sigma_y']:.1f}")
    
    # Visualisation
    fig = loader.plot_real_data(sample_data)
    plt.show()
    
    # Export
    loader.export_for_simulation(sample_data, "real_data_export.json")
