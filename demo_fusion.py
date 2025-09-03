"""
Interactive Demo for OCR-LLM Fusion Model
This script allows you to test the fused model with custom text images
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
from PIL import Image, ImageDraw, ImageFont

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
print("Loading OCR-LLM fusion model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
ocr_encoder = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1").encoder
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
brain = AutoModelForCausalLM.from_pretrained("distilgpt2")

adapter = nn.Linear(768, 768)  # OCR to LLM adapter

class OCRLLMFusion(nn.Module):
    def __init__(self, ocr_encoder, adapter, brain):
        super().__init__()
        self.ocr_encoder = ocr_encoder
        self.adapter = adapter
        self.brain = brain

fusion_model = OCRLLMFusion(ocr_encoder, adapter, brain).to(device)

# Load trained weights
try:
    fusion_model.load_state_dict(torch.load("best_ocr_llm_fusion.pth", map_location=device))
    print("✅ Loaded trained fusion model!")
except Exception as e:
    print(f"⚠️ Could not load trained weights: {e}")

fusion_model.eval()

def create_text_image(text, size=(400, 100)):
    """Create an image with the given text"""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    
    # Center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill='black', font=font)
    return img

def test_fusion_model(image_text, prompt="What text do you see?"):
    """Test the fusion model with a text image"""
    # Create test image
    test_image = create_text_image(image_text)
    
    # Process inputs
    pixel_values = processor(images=test_image, return_tensors="pt").pixel_values.to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        # Get fused embeddings
        ocr_embeds = fusion_model.ocr_encoder(pixel_values).last_hidden_state
        adapted = fusion_model.adapter(ocr_embeds)
        text_embeds = fusion_model.brain.transformer.wte(input_ids)
        
        # Fusion
        batch_size, text_seq_len, hidden_size = text_embeds.shape
        ocr_pooled = torch.mean(adapted, dim=1, keepdim=True)
        ocr_expanded = ocr_pooled.expand(batch_size, text_seq_len, hidden_size)
        inputs_embeds = text_embeds + 0.1 * ocr_expanded
        
        # Generate
        generated = fusion_model.brain.generate(
            inputs_embeds=inputs_embeds,
            max_length=input_ids.shape[1] + 20,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        result = tokenizer.decode(generated[0], skip_special_tokens=True)
        return result

# Demo
print("\n🎯 OCR-LLM Fusion Model Demo")
print("="*40)

# Test cases
test_cases = [
    "Hello AI",
    "Computer Vision",
    "Machine Learning",
    "Deep Learning",
    "Neural Network"
]

for i, text in enumerate(test_cases, 1):
    print(f"\n{i}. Testing with image text: '{text}'")
    result = test_fusion_model(text)
    print(f"   Model response: '{result}'")
    
    # Check if the model understood the image
    if text.lower() in result.lower():
        print("   ✅ SUCCESS: Model recognized the text!")
    else:
        print("   ⚠️ Model generated response but may not have fully understood the image")

print(f"\n🎉 Demo complete! The fusion model combines:")
print(f"   📸 OCR capabilities (image → text understanding)")
print(f"   🧠 LLM capabilities (text generation and reasoning)")
print(f"   🔗 Trained to fuse visual and textual information")
print(f"\nModel file: best_ocr_llm_fusion.pth ({676727611/1024/1024:.1f} MB)")
