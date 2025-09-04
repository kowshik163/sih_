"""
Enhanced Training Pipeline for FRA AI Fusion Model
Implements multi-stage training with multimodal pretraining objectives
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from PIL import Image
import rasterio
from typing import Dict, List, Tuple
import wandb
from datetime import datetime
import logging
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_fusion_model import EnhancedFRAUnifiedEncoder, MultimodalPretrainingObjectives


class EnhancedFRADataset(Dataset):
    """Enhanced dataset with multimodal pretraining support"""
    
    def __init__(self, data_path: str, config: Dict, split: str = "train"):
        self.config = config
        self.split = split
        self.mask_prob = 0.15  # Probability of masking tokens
        
        # Load training data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Split data
        split_idx = int(len(self.data) * 0.8)
        if split == "train":
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
        
        print(f"{split} dataset: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def create_positive_pairs(self, sample):
        """Create positive pairs for contrastive learning"""
        # Same village/location = positive pair
        village = sample.get('labels', {}).get('village', '')
        
        # Find another sample from same village
        for other_sample in self.data:
            if (other_sample.get('labels', {}).get('village', '') == village and 
                other_sample != sample and village != ''):
                return other_sample
        
        # If no match, return augmented version of same sample
        return sample
    
    def augment_sample(self, sample):
        """Data augmentation for multimodal data"""
        augmented = sample.copy()
        
        # Text augmentation (simple word dropout)
        if 'document_data' in sample and 'text' in sample['document_data']:
            text = sample['document_data']['text']
            words = text.split()
            if len(words) > 5:
                # Randomly drop some words
                keep_words = random.sample(words, max(1, int(len(words) * 0.8)))
                augmented['document_data']['text'] = ' '.join(keep_words)
        
        # Coordinate noise
        if 'coordinates' in sample:
            coords = sample['coordinates']
            noise = [random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)]
            augmented['coordinates'] = [coords[0] + noise[0], coords[1] + noise[1]]
        
        return augmented
    
    def create_masked_tokens(self, sample):
        """Create masked tokens for masked language modeling"""
        if 'document_data' not in sample or 'text' not in sample['document_data']:
            return sample, []
        
        text = sample['document_data']['text']
        words = text.split()
        
        masked_indices = []
        masked_words = []
        
        for i, word in enumerate(words):
            if random.random() < self.mask_prob:
                masked_indices.append(i)
                masked_words.append(word)
                words[i] = '[MASK]'
        
        masked_sample = sample.copy()
        masked_sample['document_data']['text'] = ' '.join(words)
        
        return masked_sample, masked_indices
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Create positive pair for contrastive learning
        positive_sample = self.create_positive_pairs(sample)
        
        # Create masked version
        masked_sample, masked_indices = self.create_masked_tokens(sample)
        
        # Process document data
        doc_data = sample.get('document_data', {})
        
        # Process satellite data
        sat_data = sample.get('satellite_data', {})
        satellite_image = None
        if 'tile_path' in sat_data and os.path.exists(sat_data['tile_path']):
            try:
                with rasterio.open(sat_data['tile_path']) as src:
                    bands = src.read([1, 2, 3])
                    satellite_image = np.transpose(bands, (1, 2, 0))
                    satellite_image = torch.from_numpy(satellite_image).float()
                    
                    # Resize to standard size
                    if satellite_image.shape[0] != 224 or satellite_image.shape[1] != 224:
                        satellite_image = F.interpolate(
                            satellite_image.permute(2, 0, 1).unsqueeze(0),
                            size=(224, 224),
                            mode='bilinear'
                        ).squeeze(0).permute(1, 2, 0)
            except Exception as e:
                satellite_image = torch.zeros(224, 224, 3)
        else:
            satellite_image = torch.zeros(224, 224, 3)
        
        # Process coordinates
        coordinates = sample.get('coordinates', [0.0, 0.0])
        
        # Process labels
        labels = sample.get('labels', {})
        
        return {
            'original': {
                'document_text': doc_data.get('text', ''),
                'satellite_image': satellite_image,
                'coordinates': torch.tensor(coordinates, dtype=torch.float32),
                'spectral_indices': torch.tensor([
                    sat_data.get('ndvi', 0.0),
                    sat_data.get('ndwi', 0.0),
                    sat_data.get('evi', 0.5),
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # Pad to 10 features
                ], dtype=torch.float32),
                'village_name': labels.get('village', ''),
                'status': labels.get('status', ''),
                'claim_type': labels.get('claim_type', '')
            },
            'positive': {
                'document_text': positive_sample.get('document_data', {}).get('text', ''),
                'satellite_image': satellite_image,  # Same image for now
                'coordinates': torch.tensor(positive_sample.get('coordinates', [0.0, 0.0]), dtype=torch.float32),
                'spectral_indices': torch.tensor([
                    positive_sample.get('satellite_data', {}).get('ndvi', 0.0),
                    positive_sample.get('satellite_data', {}).get('ndwi', 0.0),
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                ], dtype=torch.float32),
            },
            'masked': {
                'document_text': masked_sample.get('document_data', {}).get('text', ''),
                'masked_indices': masked_indices
            }
        }


class EnhancedFRATrainingPipeline:
    """Enhanced training pipeline with multimodal pretraining"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {self.device}")
        
        # Initialize enhanced model
        self.model = EnhancedFRAUnifiedEncoder(config['model']).to(self.device)
        
        # Initialize pretraining objectives
        self.pretraining_objectives = MultimodalPretrainingObjectives(
            temperature=config.get('model', {}).get('contrastive_learning', {}).get('temperature', 0.07)
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Setup logging
        self.setup_logging()
        
        # Initialize wandb for experiment tracking
        if config.get('use_wandb', False):
            wandb.init(project="fra-ai-fusion-enhanced", config=config)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """Prepare enhanced training and validation data loaders"""
        train_dataset = EnhancedFRADataset(data_path, self.config, split="train")
        val_dataset = EnhancedFRADataset(data_path, self.config, split="val")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=self.enhanced_collate_fn,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=self.enhanced_collate_fn,
            num_workers=2
        )
        
        return train_loader, val_loader
    
    def enhanced_collate_fn(self, batch):
        """Enhanced collate function for multimodal pretraining"""
        # Extract original samples
        original_texts = [item['original']['document_text'] for item in batch]
        satellite_images = torch.stack([item['original']['satellite_image'] for item in batch])
        coordinates = torch.stack([item['original']['coordinates'] for item in batch])
        spectral_indices = torch.stack([item['original']['spectral_indices'] for item in batch])
        
        # Extract positive samples
        positive_texts = [item['positive']['document_text'] for item in batch]
        positive_coords = torch.stack([item['positive']['coordinates'] for item in batch])
        positive_indices = torch.stack([item['positive']['spectral_indices'] for item in batch])
        
        # Extract masked samples
        masked_texts = [item['masked']['document_text'] for item in batch]
        
        # Collect labels
        village_names = [item['original']['village_name'] for item in batch]
        statuses = [item['original']['status'] for item in batch]
        claim_types = [item['original']['claim_type'] for item in batch]
        
        return {
            'original': {
                'documents': {'text': original_texts},
                'satellite_images': satellite_images.permute(0, 3, 1, 2),
                'structured_data': {
                    'latitude': coordinates[:, 0].tolist(),
                    'longitude': coordinates[:, 1].tolist(),
                    'indices': spectral_indices.tolist()
                }
            },
            'positive': {
                'documents': {'text': positive_texts},
                'structured_data': {
                    'latitude': positive_coords[:, 0].tolist(),
                    'longitude': positive_coords[:, 1].tolist(),
                    'indices': positive_indices.tolist()
                }
            },
            'masked': {
                'documents': {'text': masked_texts}
            },
            'labels': {
                'village_names': village_names,
                'statuses': statuses,
                'claim_types': claim_types
            }
        }
    
    def stage_0_multimodal_pretraining(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 15):
        """Stage 0: Multimodal pretraining with contrastive and masked objectives"""
        self.logger.info("Starting Stage 0: Multimodal Pretraining")
        
        pretraining_config = self.config['training'].get('multimodal_pretraining', {})
        contrastive_weight = pretraining_config.get('contrastive_weight', 1.0)
        masked_weight = pretraining_config.get('masked_token_weight', 0.5)
        cross_modal_weight = pretraining_config.get('cross_modal_weight', 0.3)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # Forward pass on original samples
                original_outputs = self.model(batch['original'])
                
                # Forward pass on positive samples
                positive_outputs = self.model(batch['positive'])
                
                # Forward pass on masked samples
                masked_outputs = self.model(batch['masked'])
                
                # Compute pretraining losses
                loss = 0
                
                # Contrastive loss
                if 'contrastive_embeddings' in original_outputs:
                    original_emb = original_outputs['contrastive_embeddings']
                    positive_emb = positive_outputs['contrastive_embeddings']
                    
                    # Contrastive loss between original and positive
                    contrastive_loss = self.pretraining_objectives.contrastive_loss(
                        torch.cat([original_emb, positive_emb], dim=0)
                    )
                    loss += contrastive_weight * contrastive_loss
                
                # Masked token modeling loss
                masked_indices = torch.tensor([True] * len(batch['masked']['documents']['text']))
                masked_loss = self.pretraining_objectives.masked_token_modeling(
                    masked_outputs, masked_indices
                )
                loss += masked_weight * masked_loss
                
                # Cross-modal alignment loss
                if ('contrastive_embeddings' in original_outputs and 
                    len(original_outputs['contrastive_embeddings']) > 0):
                    # Simple alignment between text and visual embeddings
                    text_emb = original_outputs['contrastive_embeddings']
                    visual_emb = original_outputs['contrastive_embeddings']  # Mock for now
                    alignment_loss = self.pretraining_objectives.cross_modal_alignment(text_emb, visual_emb)
                    loss += cross_modal_weight * alignment_loss
                
                # Backward pass
                if loss > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Stage 0 - Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Stage 0 - Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Validation
            if epoch % 2 == 0:
                val_loss = self.validate_pretraining(val_loader)
                self.logger.info(f"Stage 0 - Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        self.save_checkpoint(f"stage0_pretraining_epoch_{epochs}.pth")
        self.logger.info("Stage 0 pretraining completed")
    
    def validate_pretraining(self, val_loader: DataLoader) -> float:
        """Validation for pretraining stage"""
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                original_outputs = self.model(batch['original'])
                positive_outputs = self.model(batch['positive'])
                
                # Simple validation loss
                if 'contrastive_embeddings' in original_outputs:
                    original_emb = original_outputs['contrastive_embeddings']
                    positive_emb = positive_outputs['contrastive_embeddings']
                    val_loss = self.pretraining_objectives.contrastive_loss(
                        torch.cat([original_emb, positive_emb], dim=0)
                    )
                else:
                    val_loss = torch.tensor(0.1)
                
                total_val_loss += val_loss.item()
        
        return total_val_loss / len(val_loader)
    
    def save_checkpoint(self, filename: str):
        """Save enhanced model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(self.config['training']['checkpoint_dir'], filename)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Enhanced checkpoint saved: {checkpoint_path}")
    
    def train_enhanced_pipeline(self, data_path: str):
        """Execute enhanced training pipeline"""
        self.logger.info("Starting Enhanced FRA AI Training Pipeline")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(data_path)
        
        # Stage 0: Multimodal pretraining (NEW)
        pretraining_epochs = self.config['training']['num_epochs'].get('stage_0_pretraining', 15)
        self.stage_0_multimodal_pretraining(train_loader, val_loader, epochs=pretraining_epochs)
        
        # Continue with existing stages...
        self.logger.info("Enhanced training pipeline completed!")


def main():
    """Main enhanced training function"""
    # Enhanced training configuration
    config = {
        'model': {
            'hidden_size': 1024,
            'num_ner_labels': 15,
            'num_schemes': 50,
            'unified_token_fusion': True,
            'multimodal_pretraining': True,
            'temporal_modeling': {'enabled': True, 'd_model': 256},
            'graph_neural_network': {'enabled': True, 'hidden_dim': 512, 'num_layers': 3},
            'memory_module': {'enabled': True, 'capacity': 4096},
            'knowledge_graph': {'enabled': True, 'embedding_dim': 256},
            'contrastive_learning': {'temperature': 0.07}
        },
        'training': {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'batch_size': 2,
            'checkpoint_dir': './checkpoints',
            'num_epochs': {
                'stage_0_pretraining': 15,
                'stage_1_foundation': 10,
                'stage_2_alignment': 8,
                'stage_3_tool_skills': 5,
                'stage_4_dss': 5
            },
            'multimodal_pretraining': {
                'contrastive_weight': 1.0,
                'masked_token_weight': 0.5,
                'cross_modal_weight': 0.3
            }
        },
        'use_wandb': False
    }
    
    # Initialize enhanced trainer
    trainer = EnhancedFRATrainingPipeline(config)
    
    # Start enhanced training
    data_path = "../1_data_processing/training_data.json"
    
    if os.path.exists(data_path):
        trainer.train_enhanced_pipeline(data_path)
    else:
        print(f"Training data not found at {data_path}")
        print("Please run the data processing pipeline first")


if __name__ == "__main__":
    main()
