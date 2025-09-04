# ========================================
# Step 1. Install libraries
# ========================================
# !pip install torch torchvision transformers datasets accelerate timm pillow requests

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from PIL import Image
import requests
import os
from torch.nn.utils.rnn import pad_sequence

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================================
# Step 2. Load OCR Encoder (Eyes)
# ========================================
print("Loading OCR encoder...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
ocr_encoder = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1").encoder
ocr_hidden = ocr_encoder.config.hidden_size

# ========================================
# Step 3. Load LLM (Brain)
# ========================================
print("Loading DistilGPT2...")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad by default
brain = AutoModelForCausalLM.from_pretrained("distilgpt2")
llm_hidden = brain.config.hidden_size

# ========================================
# Step 4. Add Translator (Adapter)
# ========================================
print("Creating adapter...")
adapter = nn.Linear(ocr_hidden, llm_hidden)

# Wrap everything into one model
class OCRLLMFusion(nn.Module):
    def __init__(self, ocr_encoder, adapter, brain):
        super().__init__()
        self.ocr_encoder = ocr_encoder
        self.adapter = adapter
        self.brain = brain

    def forward(self, pixel_values, input_ids, labels=None):
        # 1. Extract OCR embeddings
        with torch.no_grad():  # keep encoder frozen at first
            ocr_embeds = self.ocr_encoder(pixel_values).last_hidden_state
        adapted = self.adapter(ocr_embeds)  # [batch_size, ocr_seq_len, hidden_size]

        # 2. Get text embeddings
        text_embeds = self.brain.transformer.wte(input_ids)  # [batch_size, text_seq_len, hidden_size]

        # 3. Pool OCR embeddings to match text sequence length or use mean pooling
        batch_size, text_seq_len, hidden_size = text_embeds.shape
        
        # Mean pool the OCR features to a single vector and expand to match text length
        ocr_pooled = torch.mean(adapted, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
        ocr_expanded = ocr_pooled.expand(batch_size, text_seq_len, hidden_size)  # [batch_size, text_seq_len, hidden_size]
        
        # 4. Add OCR features to text embeddings instead of concatenating
        inputs_embeds = text_embeds + 0.1 * ocr_expanded  # Weighted addition

        # 5. Forward through LLM with proper label alignment
        if labels is not None:
            # Ensure labels match the input sequence length
            labels = labels[:, :text_seq_len]
        
        outputs = self.brain(inputs_embeds=inputs_embeds, labels=labels)
        return outputs

fusion_model = OCRLLMFusion(ocr_encoder, adapter, brain).to(device)

# ========================================
# Step 5. Custom Dataset for Multiple OCR Sources
# ========================================
class MultiOCRDataset(Dataset):
    def __init__(self, processor, tokenizer, max_length=128):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load different datasets
        print("Loading FUNSD dataset...")
        try:
            funsd = load_dataset("nielsr/funsd-layoutlmv3", split="train[:100]")
            for item in funsd:
                if item["image"] is not None and item["tokens"]:
                    text = " ".join([token for token in item["tokens"] if token.strip()])
                    if text.strip():
                        self.data.append({
                            "image": item["image"],
                            "text": text[:200]  # Truncate long texts
                        })
        except Exception as e:
            print(f"Error loading FUNSD: {e}")
        
        # Add synthetic text examples (simulating other datasets)
        print("Adding synthetic examples...")
        try:
            # Create some dummy examples for demo
            dummy_texts = [
                "Hello world this is a test",
                "OCR model training data",
                "Computer vision and natural language processing",
                "Scene text recognition example",
                "Document understanding task"
            ]
            
            # Create dummy images (white background with text)
            for text in dummy_texts:
                # Create a simple white image (placeholder)
                img = Image.new('RGB', (224, 224), color='white')
                self.data.append({
                    "image": img,
                    "text": text
                })
        except Exception as e:
            print(f"Error creating synthetic data: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process image
        try:
            if isinstance(item["image"], str):
                image = Image.open(item["image"]).convert("RGB")
            else:
                image = item["image"].convert("RGB")
        except:
            # Create dummy image if error
            image = Image.new('RGB', (224, 224), color='white')
        
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Process text
        text = item["text"]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0),
            "labels": tokens.input_ids.squeeze(0)
        }

# Create datasets
print("Creating datasets...")
train_dataset = MultiOCRDataset(processor, tokenizer)
val_dataset = MultiOCRDataset(processor, tokenizer)  # Using same for simplicity

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Custom collate function
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)

# ========================================
# Step 6. Training Loop with Improved Error Handling
# ========================================
print("Starting training...")

optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

num_epochs = 2
best_loss = float('inf')

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
    
    # Training phase
    fusion_model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Use the actual input_ids instead of dummy ones
            optimizer.zero_grad()
            
            outputs = fusion_model(pixel_values, input_ids, labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / max(num_batches, 1)
    print(f"Average training loss: {avg_loss:.4f}")
    
    # Validation phase
    fusion_model.eval()
    val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = fusion_model(pixel_values, input_ids, labels)
                val_loss += outputs.loss.item()
                val_batches += 1
                
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
    
    avg_val_loss = val_loss / max(val_batches, 1)
    print(f"Average validation loss: {avg_val_loss:.4f}")
    
    # Save best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        print(f"New best validation loss: {best_loss:.4f}")
        torch.save(fusion_model.state_dict(), "best_ocr_llm_fusion.pth")
    
    scheduler.step()

print("\n=== Training Complete ===")

# ========================================
# Step 7. Test Inference
# ========================================
print("\n=== Testing Inference ===")
fusion_model.eval()

# Get a sample from validation dataset
try:
    sample_batch = next(iter(val_loader))
    pixel_values = sample_batch["pixel_values"][:1].to(device)  # Take first sample
    input_ids = sample_batch["input_ids"][:1].to(device)
    
    print("Original text:", tokenizer.decode(input_ids[0], skip_special_tokens=True))
    
    with torch.no_grad():
        outputs = fusion_model(pixel_values, input_ids)
        
        # Generate text using the brain model directly with proper inputs_embeds
        # Get the fused embeddings from our forward pass
        ocr_embeds = fusion_model.ocr_encoder(pixel_values).last_hidden_state
        adapted = fusion_model.adapter(ocr_embeds)
        text_embeds = fusion_model.brain.transformer.wte(input_ids)
        
        # Mean pool and add OCR features
        batch_size, text_seq_len, hidden_size = text_embeds.shape
        ocr_pooled = torch.mean(adapted, dim=1, keepdim=True)
        ocr_expanded = ocr_pooled.expand(batch_size, text_seq_len, hidden_size)
        inputs_embeds = text_embeds + 0.1 * ocr_expanded
        
        generated = brain.generate(
            inputs_embeds=inputs_embeds,
            max_length=input_ids.shape[1] + 20,  # Generate a bit more
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print("Generated text:", generated_text)

except Exception as e:
    print(f"Error during inference: {e}")

print("\n=== Model Fusion Complete ===")
print(f"Model saved as: best_ocr_llm_fusion.pth")
print(f"Device used: {device}")
print(f"OCR encoder hidden size: {ocr_hidden}")
print(f"LLM hidden size: {llm_hidden}")