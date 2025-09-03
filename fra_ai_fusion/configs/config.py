"""
Main Configuration File for FRA AI Fusion System
Centralized configuration management
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import json

class FRAConfig:
    """Main configuration class for FRA AI Fusion System"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent.parent
        self.config_path = config_path or self.base_dir / "configs" / "config.json"
        
        # Load configuration
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Create default configuration
            default_config = self.get_default_config()
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: Dict):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            # Model Configuration
            "model": {
                "name": "FRA_Unified_AI",
                "version": "1.0.0",
                "hidden_size": 1024,
                "num_ner_labels": 15,
                "num_schemes": 50,
                "max_sequence_length": 512,
                "temperature": 0.7,
                "device": "auto"  # auto, cpu, cuda
            },
            
            # Training Configuration
            "training": {
                "learning_rate": 1e-4,
                "batch_size": 4,
                "num_epochs": {
                    "stage_1_foundation": 10,
                    "stage_2_alignment": 8,
                    "stage_3_tool_skills": 5,
                    "stage_4_dss": 5
                },
                "weight_decay": 0.01,
                "gradient_clip": 1.0,
                "warmup_steps": 1000,
                "save_every_n_epochs": 2,
                "validate_every_n_epochs": 1,
                "checkpoint_dir": "./checkpoints",
                "log_dir": "./logs"
            },
            
            # Data Configuration
            "data": {
                "raw_data_dir": "./data/raw",
                "processed_data_dir": "./data/processed",
                "training_data_path": "./data/training_data.json",
                "validation_split": 0.2,
                "test_split": 0.1,
                "max_image_size": [224, 224],
                "supported_formats": {
                    "documents": [".pdf", ".png", ".jpg", ".jpeg", ".tiff"],
                    "satellite": [".tif", ".tiff", ".jp2"],
                    "vector": [".shp", ".geojson", ".gpkg"]
                }
            },
            
            # Database Configuration
            "database": {
                "type": "postgresql",
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "name": os.getenv("POSTGRES_DB", "fra_gis"),
                "user": os.getenv("POSTGRES_USER", "fra_user"),
                "password": os.getenv("POSTGRES_PASSWORD", "fra_password"),
                "pool_size": 10,
                "max_overflow": 20
            },
            
            # API Configuration
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": True,
                "cors_origins": ["*"],
                "max_upload_size": 50 * 1024 * 1024,  # 50MB
                "rate_limit": {
                    "requests_per_minute": 100,
                    "burst_size": 20
                }
            },
            
            # External Services
            "external_services": {
                "geoserver": {
                    "url": "http://localhost:8080/geoserver",
                    "username": "admin",
                    "password": "geoserver",
                    "workspace": "fra_workspace"
                },
                "tile_servers": {
                    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                    "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                }
            },
            
            # Processing Configuration
            "processing": {
                "ocr": {
                    "engine": "tesseract",
                    "languages": ["eng", "hin"],
                    "confidence_threshold": 30,
                    "preprocessing": {
                        "denoise": True,
                        "contrast_enhancement": True,
                        "skew_correction": True
                    }
                },
                "satellite": {
                    "bands": {
                        "sentinel2": ["B2", "B3", "B4", "B8", "B11", "B12"],
                        "landsat8": ["B2", "B3", "B4", "B5", "B6", "B7"]
                    },
                    "indices": ["NDVI", "NDWI", "EVI", "SAVI"],
                    "cloud_threshold": 20  # Percentage
                },
                "parallel": {
                    "num_workers": 4,
                    "batch_size": 16
                }
            },
            
            # NER Labels Configuration
            "ner_labels": {
                "village_name": 0,
                "patta_holder": 1,
                "claim_type": 2,
                "status": 3,
                "coordinates": 4,
                "area": 5,
                "date": 6,
                "district": 7,
                "state": 8,
                "survey_number": 9,
                "revenue_village": 10,
                "block": 11,
                "tehsil": 12,
                "forest_range": 13,
                "other": 14
            },
            
            # CSS Schemes Configuration
            "css_schemes": {
                "PM-KISAN": {
                    "id": 0,
                    "ministry": "Agriculture & Farmers Welfare",
                    "beneficiary_type": "Individual",
                    "amount": 6000,
                    "frequency": "Annual"
                },
                "Jal Jeevan Mission": {
                    "id": 1,
                    "ministry": "Jal Shakti",
                    "beneficiary_type": "Household",
                    "amount": "Variable",
                    "frequency": "One-time"
                },
                "MGNREGA": {
                    "id": 2,
                    "ministry": "Rural Development",
                    "beneficiary_type": "Individual",
                    "amount": "Variable",
                    "frequency": "Daily wages"
                },
                "DAJGUA": {
                    "id": 3,
                    "ministry": "Multiple",
                    "beneficiary_type": "Community",
                    "amount": "Variable",
                    "frequency": "Project-based"
                }
            },
            
            # Land Use Classification
            "land_use_classes": {
                "forest": 1,
                "agriculture": 2,
                "water": 3,
                "built_up": 4,
                "barren": 5,
                "other": 0
            },
            
            # Logging Configuration
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "./logs/fra_fusion.log",
                "max_size": "10MB",
                "backup_count": 5
            },
            
            # Security Configuration
            "security": {
                "jwt_secret": os.getenv("JWT_SECRET", "your-secret-key-here"),
                "jwt_expire_hours": 24,
                "allowed_ips": [],  # Empty = allow all
                "encrypt_sensitive_data": True
            },
            
            # Performance Configuration
            "performance": {
                "cache": {
                    "enabled": True,
                    "type": "redis",  # redis, memory
                    "ttl": 3600,  # seconds
                    "max_size": 1000
                },
                "model_optimization": {
                    "quantization": True,
                    "pruning": False,
                    "onnx_export": False
                }
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
        
        # Save updated configuration
        self.save_config(self.config)
    
    def update(self, updates: Dict):
        """Update configuration with dictionary"""
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        deep_update(self.config, updates)
        self.save_config(self.config)
    
    @property
    def model_config(self) -> Dict:
        return self.config["model"]
    
    @property
    def training_config(self) -> Dict:
        return self.config["training"]
    
    @property
    def data_config(self) -> Dict:
        return self.config["data"]
    
    @property
    def database_config(self) -> Dict:
        return self.config["database"]
    
    @property
    def api_config(self) -> Dict:
        return self.config["api"]
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        db_config = self.database_config
        return f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}"
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check required directories
        for dir_key in ["checkpoint_dir", "log_dir"]:
            path = Path(self.get(f"training.{dir_key}"))
            if not path.parent.exists():
                errors.append(f"Parent directory for {dir_key} does not exist: {path.parent}")
        
        # Check model configuration
        if self.get("model.hidden_size", 0) <= 0:
            errors.append("Model hidden_size must be positive")
        
        # Check training configuration
        if self.get("training.learning_rate", 0) <= 0:
            errors.append("Learning rate must be positive")
        
        # Check batch size
        if self.get("training.batch_size", 0) <= 0:
            errors.append("Batch size must be positive")
        
        return errors


# Global configuration instance
config = FRAConfig()

# Convenience functions
def get_config(key: str, default=None):
    """Get configuration value"""
    return config.get(key, default)

def set_config(key: str, value):
    """Set configuration value"""
    config.set(key, value)

def validate_config() -> List[str]:
    """Validate current configuration"""
    return config.validate_config()

def get_database_url() -> str:
    """Get database connection URL"""
    return config.get_database_url()

# Export main configuration object
__all__ = ['config', 'get_config', 'set_config', 'validate_config', 'get_database_url']
