
# src/models/model_registry.py

"""
Model Registry

Handles saving, loading, and versioning of trained models.
Ensures reproducibility and easy deployment.
"""

import os
import json
import joblib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing trained model artifacts.
    
    Provides:
    - Versioned model storage
    - Metadata tracking (metrics, parameters, training date)
    - Easy loading of latest or specific model version
    - Model comparison across versions
    
    Directory structure:
        models/
        ├── registry.json           # Master registry file
        ├── v1/
        │   ├── xgboost/
        │   │   ├── model.joblib
        │   │   └── metadata.json
        │   ├── prophet/
        │   │   ├── model.joblib
        │   │   └── metadata.json
        │   ├── random_forest/
        │   │   ├── model.joblib
        │   │   └── metadata.json
        │   ├── ensemble/
        │   │   ├── ensemble_config.json
        │   │   └── meta_model.joblib
        │   ├── scalers/
        │   │   └── scaler.joblib
        │   └── encoders/
        │       └── label_encoders.joblib
        └── v2/
            └── ...
    """
    
    def __init__(self, base_dir: str = 'models'):
        """
        Initialize ModelRegistry.
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = base_dir
        self.registry_path = os.path.join(base_dir, 'registry.json')
        
        os.makedirs(base_dir, exist_ok=True)
        
        # Load or create registry
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                'versions': [],
                'latest_version': None,
                'best_version': None,
                'created_at': datetime.now().isoformat()
            }
    
    def register_version(
        self,
        models: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
        parameters: Dict[str, Any],
        scalers: Optional[Dict[str, Any]] = None,
        encoders: Optional[Dict[str, Any]] = None,
        ensemble: Optional[Any] = None,
        description: str = ""
    ) -> str:
        """
        Register a new model version.
        
        Args:
            models: Dict of {model_name: trained_model_object}
            metrics: Dict of {model_name: {metric: value}}
            parameters: Training parameters used
            scalers: Optional preprocessing scalers
            encoders: Optional label/target encoders
            ensemble: Optional fitted ensemble object
            description: Human-readable description of this version
            
        Returns:
            Version string (e.g., 'v1', 'v2')
        """
        # Determine version number
        version_num = len(self.registry['versions']) + 1
        version = f"v{version_num}"
        version_dir = os.path.join(self.base_dir, version)
        
        logger.info(f"Registering model version: {version}")
        
        # Save individual models
        for name, model in models.items():
            model_dir = os.path.join(version_dir, name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            joblib.dump(
                model,
                os.path.join(model_dir, 'model.joblib')
            )
            
            # Save model metadata
            metadata = {
                'model_name': name,
                'version': version,
                'metrics': metrics.get(name, {}),
                'saved_at': datetime.now().isoformat()
            }
            with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"  Saved {name} to {model_dir}")
        
        # Save scalers
        if scalers:
            scaler_dir = os.path.join(version_dir, 'scalers')
            os.makedirs(scaler_dir, exist_ok=True)
            for name, scaler in scalers.items():
                joblib.dump(
                    scaler,
                    os.path.join(scaler_dir, f'{name}.joblib')
                )
            logger.info(f"  Saved {len(scalers)} scalers")
        
        # Save encoders
        if encoders:
            encoder_dir = os.path.join(version_dir, 'encoders')
            os.makedirs(encoder_dir, exist_ok=True)
            for name, encoder in encoders.items():
                joblib.dump(
                    encoder,
                    os.path.join(encoder_dir, f'{name}.joblib')
                )
            logger.info(f"  Saved {len(encoders)} encoders")
        
        # Save ensemble
        if ensemble:
            ensemble_dir = os.path.join(version_dir, 'ensemble')
            ensemble.save(ensemble_dir)
            logger.info(f"  Saved ensemble to {ensemble_dir}")
        
        # Get ensemble metric for version-level comparison
        ensemble_rmse = None
        if 'ensemble' in metrics:
            ensemble_rmse = metrics['ensemble'].get('rmse', None)
        elif ensemble and hasattr(ensemble, 'ensemble_metrics'):
            ensemble_rmse = ensemble.ensemble_metrics.get('rmse', None)
        
        # Update registry
        version_entry = {
            'version': version,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'models': list(models.keys()),
            'metrics': {
                k: {
                    mk: round(mv, 6) if isinstance(mv, float) else mv
                    for mk, mv in v.items()
                }
                for k, v in metrics.items()
            },
            'ensemble_rmse': ensemble_rmse,
            'parameters_hash': str(hash(str(parameters)))
        }
        
        self.registry['versions'].append(version_entry)
        self.registry['latest_version'] = version
        
        # Update best version (lowest ensemble RMSE)
        if ensemble_rmse is not None:
            if (
                self.registry['best_version'] is None or
                ensemble_rmse < self._get_version_rmse(
                    self.registry['best_version']
                )
            ):
                self.registry['best_version'] = version
                logger.info(f"  🏆 New best version: {version} "
                           f"(RMSE: {ensemble_rmse:.4f})")
        
        # Save registry
        self._save_registry()
        
        logger.info(f"✅ Version {version} registered successfully")
        
        return version
    
    def load_version(
        self,
        version: Optional[str] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Load a specific model version.
        
        Args:
            version: Version to load (e.g., 'v1'). If None, loads latest.
            load_best: If True, load the best-performing version
            
        Returns:
            Dictionary with loaded models, scalers, encoders, ensemble
        """
        if load_best:
            version = self.registry['best_version']
        elif version is None:
            version = self.registry['latest_version']
        
        if version is None:
            raise ValueError("No versions registered yet")
        
        version_dir = os.path.join(self.base_dir, version)
        
        if not os.path.exists(version_dir):
            raise FileNotFoundError(f"Version directory not found: {version_dir}")
        
        logger.info(f"Loading model version: {version}")
        
        loaded = {
            'version': version,
            'models': {},
            'scalers': {},
            'encoders': {},
            'ensemble': None
        }
        
        # Load individual models
        for item in os.listdir(version_dir):
            item_path = os.path.join(version_dir, item)
            
            if not os.path.isdir(item_path):
                continue
            
            if item == 'scalers':
                for f in os.listdir(item_path):
                    if f.endswith('.joblib'):
                        name = f.replace('.joblib', '')
                        loaded['scalers'][name] = joblib.load(
                            os.path.join(item_path, f)
                        )
                        
            elif item == 'encoders':
                for f in os.listdir(item_path):
                    if f.endswith('.joblib'):
                        name = f.replace('.joblib', '')
                        loaded['encoders'][name] = joblib.load(
                            os.path.join(item_path, f)
                        )
                        
            elif item == 'ensemble':
                from src.models.ensemble_model import EnsembleForecaster
                loaded['ensemble'] = EnsembleForecaster.load(item_path)
                
            else:
                model_path = os.path.join(item_path, 'model.joblib')
                if os.path.exists(model_path):
                    loaded['models'][item] = joblib.load(model_path)
                    logger.info(f"  Loaded model: {item}")
        
        logger.info(f"✅ Loaded version {version}: "
                    f"{len(loaded['models'])} models, "
                    f"{len(loaded['scalers'])} scalers, "
                    f"{len(loaded['encoders'])} encoders, "
                    f"ensemble={'yes' if loaded['ensemble'] else 'no'}")
        
        return loaded
    
    def compare_versions(self) -> pd.DataFrame:
        """
        Compare all registered versions.
        
        Returns DataFrame with version-level metrics for comparison.
        """
        import pandas as pd
        
        rows = []
        for v in self.registry['versions']:
            row = {
                'version': v['version'],
                'created_at': v['created_at'],
                'description': v['description'],
                'n_models': len(v['models']),
                'ensemble_rmse': v.get('ensemble_rmse', None)
            }
            
            # Add individual model metrics
            for model_name, model_metrics in v.get('metrics', {}).items():
                for metric_name, metric_value in model_metrics.items():
                    row[f'{model_name}_{metric_name}'] = metric_value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if 'ensemble_rmse' in df.columns:
            df = df.sort_values('ensemble_rmse')
        
        return df
    
    def _get_version_rmse(self, version: str) -> float:
        """Get the ensemble RMSE for a specific version."""
        for v in self.registry['versions']:
            if v['version'] == version:
                return v.get('ensemble_rmse', float('inf'))
        return float('inf')
    
    def _save_registry(self):
        """Save registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
