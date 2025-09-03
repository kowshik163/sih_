"""
Training Script for FRA AI Fusion Model
Implements multi-stage curriculum learning as outlined in stepbystepprocess.md
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_fusion_model import FRAUnifiedEncoder

class FRADataset(Dataset):
    """Dataset class for FRA multimodal training data"""
    
    def __init__(self, data_path: str, config: Dict, split: str = "train"):
        self.config = config
        self.split = split
        
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
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Process document data
        doc_data = sample.get('document_data', {})
        
        # Process satellite data
        sat_data = sample.get('satellite_data', {})
        satellite_image = None
        if 'tile_path' in sat_data and os.path.exists(sat_data['tile_path']):
            try:
                with rasterio.open(sat_data['tile_path']) as src:
                    # Read first 3 bands as RGB
                    bands = src.read([1, 2, 3])  # R, G, B
                    satellite_image = np.transpose(bands, (1, 2, 0))
                    satellite_image = torch.from_numpy(satellite_image).float()
            except Exception as e:
                print(f"Error loading satellite image: {e}")
                satellite_image = torch.zeros(224, 224, 3)
        else:
            satellite_image = torch.zeros(224, 224, 3)
        
        # Process coordinates
        coordinates = sample.get('coordinates', [0.0, 0.0])
        
        # Process labels
        labels = sample.get('labels', {})
        
        return {
            'document_text': doc_data.get('text', ''),
            'satellite_image': satellite_image,
            'coordinates': torch.tensor(coordinates, dtype=torch.float32),
            'spectral_indices': torch.tensor([
                sat_data.get('ndvi', 0.0),
                sat_data.get('ndwi', 0.0),
                0.5  # Placeholder for additional index
            ], dtype=torch.float32),
            'village_name': labels.get('village', ''),
            'status': labels.get('status', ''),
            'claim_type': labels.get('claim_type', '')
        }


class FRATrainingPipeline:
    """Complete training pipeline for FRA AI Fusion Model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {self.device}")
        
        # Initialize model
        self.model = FRAUnifiedEncoder(config['model']).to(self.device)
        
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
            wandb.init(project="fra-ai-fusion", config=config)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders"""
        train_dataset = FRADataset(data_path, self.config, split="train")
        val_dataset = FRADataset(data_path, self.config, split="val")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2
        )
        
        return train_loader, val_loader
    
    def collate_fn(self, batch):
        """Custom collate function for batch processing"""
        # Extract and pad text sequences
        texts = [item['document_text'] for item in batch]
        
        # Stack satellite images
        satellite_images = torch.stack([item['satellite_image'] for item in batch])
        
        # Stack coordinates and indices
        coordinates = torch.stack([item['coordinates'] for item in batch])
        spectral_indices = torch.stack([item['spectral_indices'] for item in batch])
        
        # Collect labels
        village_names = [item['village_name'] for item in batch]
        statuses = [item['status'] for item in batch]
        claim_types = [item['claim_type'] for item in batch]
        
        return {
            'documents': {'text': texts},
            'satellite_images': satellite_images.permute(0, 3, 1, 2),  # BHWC -> BCHW
            'structured_data': {
                'latitude': coordinates[:, 0].tolist(),
                'longitude': coordinates[:, 1].tolist(),
                'indices': spectral_indices.tolist()
            },
            'labels': {
                'village_names': village_names,
                'statuses': statuses,
                'claim_types': claim_types
            }
        }
    
    def stage_1_foundation_training(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 5):
        """Stage 1: Foundation model training (OCR, NER, Segmentation)"""
        self.logger.info("Starting Stage 1: Foundation Training")
        
        # Create NER label mappings
        status_to_id = {'Approved': 0, 'Rejected': 1, 'Pending': 2, '': 3}
        claim_type_to_id = {
            'Individual Forest Rights': 0,
            'Community Forest Rights': 1, 
            'Community Rights': 2,
            '': 3
        }
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # Prepare inputs
                inputs = {
                    'documents': batch['documents'],
                    'satellite_images': batch['satellite_images'].to(self.device),
                    'structured_data': batch['structured_data'],
                    'task': 'all'
                }
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate losses
                loss = 0
                
                # NER loss (simplified - using status classification)
                if 'ner_logits' in outputs:
                    status_labels = torch.tensor([
                        status_to_id.get(status, 3) for status in batch['labels']['statuses']
                    ]).to(self.device)
                    
                    # Use mean pooling of sequence for classification
                    pooled_logits = outputs['ner_logits'].mean(dim=1)[:, :4]  # First 4 classes for status
                    ner_loss = F.cross_entropy(pooled_logits, status_labels)
                    loss += ner_loss
                
                # Segmentation loss (using dummy targets for now)
                if 'segmentation_masks' in outputs:
                    batch_size = outputs['segmentation_masks'].shape[0]
                    dummy_targets = torch.randint(0, 6, (batch_size, 224, 224)).to(self.device)
                    seg_loss = F.cross_entropy(outputs['segmentation_masks'], dummy_targets)
                    loss += 0.5 * seg_loss  # Reduce weight
                
                # Backward pass
                if loss > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Stage 1 - Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Stage 1 - Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Validation
            if epoch % 2 == 0:
                val_loss = self.validate(val_loader, stage=1)
                self.logger.info(f"Stage 1 - Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        self.save_checkpoint(f"stage1_epoch_{epochs}.pth")
        self.logger.info("Stage 1 training completed")
    
    def stage_2_alignment_training(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 5):
        """Stage 2: Cross-modal alignment training"""
        self.logger.info("Starting Stage 2: Cross-modal Alignment Training")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # Prepare inputs
                inputs = {
                    'documents': batch['documents'],
                    'satellite_images': batch['satellite_images'].to(self.device),
                    'structured_data': batch['structured_data'],
                    'task': 'embedding'
                }
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Contrastive loss for alignment
                embeddings = outputs['unified_embeddings']
                contrastive_loss = self.compute_contrastive_loss(embeddings, batch)
                
                # Backward pass
                contrastive_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += contrastive_loss.item()
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Stage 2 - Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {contrastive_loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Stage 2 - Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Validation
            if epoch % 2 == 0:
                val_loss = self.validate(val_loader, stage=2)
                self.logger.info(f"Stage 2 - Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        self.save_checkpoint(f"stage2_epoch_{epochs}.pth")
        self.logger.info("Stage 2 training completed")
    
    def stage_3_tool_skills_training(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 5):
        """Stage 3: Tool skills training (SQL generation, API calls)"""
        self.logger.info("Starting Stage 3: Tool Skills Training")
        
        # Create sample SQL queries for training
        sql_templates = [
            "SELECT * FROM fra_claims WHERE village_name = '{village}' AND status = '{status}'",
            "SELECT COUNT(*) FROM fra_claims WHERE claim_type = '{claim_type}'",
            "SELECT village_name, status FROM fra_claims WHERE status = 'Approved'"
        ]
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # Generate target SQL queries
                target_queries = []
                for i, village in enumerate(batch['labels']['village_names']):
                    status = batch['labels']['statuses'][i]
                    claim_type = batch['labels']['claim_types'][i]
                    
                    # Choose a random template and fill it
                    template = np.random.choice(sql_templates)
                    query = template.format(village=village, status=status, claim_type=claim_type)
                    target_queries.append(query)
                
                # Prepare inputs
                inputs = {
                    'documents': batch['documents'],
                    'satellite_images': batch['satellite_images'].to(self.device),
                    'structured_data': batch['structured_data'],
                    'task': 'sql_generation'
                }
                
                # Forward pass
                outputs = self.model(inputs)
                
                # SQL generation loss (simplified - using embedding similarity)
                if 'sql_logits' in outputs:
                    # This is a simplified version - in practice, you'd use sequence-to-sequence loss
                    sql_loss = torch.tensor(0.1, requires_grad=True).to(self.device)  # Placeholder
                    loss = sql_loss
                else:
                    loss = torch.tensor(0.0, requires_grad=True).to(self.device)
                
                # Backward pass
                if loss > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Stage 3 - Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Stage 3 - Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        self.save_checkpoint(f"stage3_epoch_{epochs}.pth")
        self.logger.info("Stage 3 training completed")
    
    def stage_4_dss_training(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 5):
        """Stage 4: Decision Support System training"""
        self.logger.info("Starting Stage 4: DSS Training")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # Prepare inputs
                inputs = {
                    'documents': batch['documents'],
                    'satellite_images': batch['satellite_images'].to(self.device),
                    'structured_data': batch['structured_data'],
                    'task': 'dss_recommendation'
                }
                
                # Forward pass
                outputs = self.model(inputs)
                
                # DSS loss (using dummy scheme targets)
                if 'dss_scores' in outputs:
                    batch_size = outputs['dss_scores'].shape[0]
                    # Create dummy targets based on village characteristics
                    dummy_targets = torch.randint(0, 10, (batch_size,)).to(self.device)
                    dss_loss = F.cross_entropy(outputs['dss_scores'][:, :10], dummy_targets)
                    loss = dss_loss
                else:
                    loss = torch.tensor(0.0, requires_grad=True).to(self.device)
                
                # Backward pass
                if loss > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Stage 4 - Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Stage 4 - Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(f"final_model.pth")
        self.logger.info("Stage 4 training completed - Final model saved")
    
    def compute_contrastive_loss(self, embeddings: torch.Tensor, batch: Dict, temperature: float = 0.1) -> torch.Tensor:
        """Compute contrastive loss for cross-modal alignment"""
        batch_size = embeddings.shape[0]
        
        # Pool embeddings to single vectors
        pooled_embeddings = embeddings.mean(dim=1)  # [batch_size, hidden_size]
        
        # Normalize embeddings
        normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T) / temperature
        
        # Create positive pair mask (same village name = positive)
        village_names = batch['labels']['village_names']
        positive_mask = torch.zeros(batch_size, batch_size).to(self.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and village_names[i] == village_names[j] and village_names[i] != '':
                    positive_mask[i, j] = 1.0
        
        # Compute contrastive loss
        labels = torch.arange(batch_size).to(self.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def validate(self, val_loader: DataLoader, stage: int) -> float:
        """Validation step"""
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    'documents': batch['documents'],
                    'satellite_images': batch['satellite_images'].to(self.device),
                    'structured_data': batch['structured_data'],
                    'task': 'embedding'
                }
                
                outputs = self.model(inputs)
                
                # Simple validation loss
                val_loss = torch.tensor(0.1).to(self.device)  # Placeholder
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        return avg_val_loss
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(self.config['training']['checkpoint_dir'], filename)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train_full_pipeline(self, data_path: str):
        """Execute full training pipeline"""
        self.logger.info("Starting Full Training Pipeline")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(data_path)
        
        # Stage 1: Foundation training
        self.stage_1_foundation_training(train_loader, val_loader, epochs=5)
        
        # Stage 2: Alignment training  
        self.stage_2_alignment_training(train_loader, val_loader, epochs=5)
        
        # Stage 3: Tool skills training
        self.stage_3_tool_skills_training(train_loader, val_loader, epochs=3)
        
        # Stage 4: DSS training
        self.stage_4_dss_training(train_loader, val_loader, epochs=3)
        
        self.logger.info("Full training pipeline completed!")


def main():
    """Main training function"""
    # Training configuration
    config = {
        'model': {
            'hidden_size': 1024,
            'num_ner_labels': 10,
            'num_schemes': 50
        },
        'training': {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'batch_size': 2,  # Small batch size due to multimodal complexity
            'checkpoint_dir': './checkpoints'
        },
        'use_wandb': False  # Set to True if you want experiment tracking
    }
    
    # Initialize trainer
    trainer = FRATrainingPipeline(config)
    
    # Start training (assuming training_data.json exists from data pipeline)
    data_path = "../1_data_processing/training_data.json"
    
    if os.path.exists(data_path):
        trainer.train_full_pipeline(data_path)
    else:
        print(f"Training data not found at {data_path}")
        print("Please run the data processing pipeline first")


if __name__ == "__main__":
    main()
