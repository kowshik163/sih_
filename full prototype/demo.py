#!/usr/bin/env python3
"""
FRA AI Fusion System - Quick Demo
Demonstrates the unified AI model capabilities
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from main_fusion_model import FRAUnifiedEncoder
from configs.config import config

def demo_unified_model():
    """Demonstrate the unified model capabilities"""
    print("🌲 FRA AI Fusion System - Quick Demo 🌲")
    print("=" * 50)
    
    # Initialize the unified model
    print("🧠 Initializing Unified AI Model...")
    model_config = {
        'hidden_size': 1024,
        'num_ner_labels': 10,
        'num_schemes': 50
    }
    
    model = FRAUnifiedEncoder(model_config)
    print("✅ Model initialized successfully!")
    
    # Demo 1: Document Processing + Satellite Analysis + Geospatial Understanding
    print("\n📄 Demo 1: Multimodal Understanding")
    sample_input = {
        "documents": {
            "text": "Village: Adilabad, Patta Holder: Raman Singh, Claim Type: IFR, Status: Approved, Area: 2.5 hectares"
        },
        "structured_data": {
            "latitude": [18.6722],
            "longitude": [79.0011],
            "indices": [0.7, 0.3, 0.8]  # High NDVI (forest), Low NDWI (dry), High forest cover
        },
        "task": "all"
    }
    
    print(f"Input: {sample_input['documents']['text']}")
    print(f"Coordinates: {sample_input['structured_data']['latitude'][0]}, {sample_input['structured_data']['longitude'][0]}")
    
    with torch.no_grad():
        outputs = model(sample_input)
        print(f"✅ Unified embeddings shape: {outputs['unified_embeddings'].shape}")
        print(f"✅ Generated embeddings for document + geospatial + spectral data")
    
    # Demo 2: Natural Language to SQL Generation
    print("\n🗣️ Demo 2: Natural Language Query Processing")
    queries = [
        "Show all approved FRA claims in Telangana",
        "Find villages with high forest cover and pending claims",
        "List community forest rights in Odisha approved after 2020"
    ]
    
    for query in queries:
        sql = model.generate_sql(query)
        print(f"Query: {query}")
        print(f"Generated SQL: {sql[:100]}...")
        print()
    
    # Demo 3: Decision Support System
    print("\n💡 Demo 3: AI-Powered Scheme Recommendations")
    village_scenarios = [
        {
            "name": "Forest Village (High NDVI)",
            "data": {
                "latitude": [18.5],
                "longitude": [79.0],
                "indices": [0.8, 0.2, 0.9],  # High forest, low water
                "census_data": [800, 600, 0.5]  # Small, mostly tribal, low literacy
            }
        },
        {
            "name": "Agricultural Village (Medium NDVI)", 
            "data": {
                "latitude": [20.2],
                "longitude": [78.5],
                "indices": [0.4, 0.6, 0.3],  # Agriculture, high water, less forest
                "census_data": [1500, 400, 0.75]  # Larger, mixed, higher literacy
            }
        }
    ]
    
    for scenario in village_scenarios:
        print(f"Scenario: {scenario['name']}")
        recommendations = model.recommend_schemes(scenario['data'])
        
        print("Top 3 Recommended Schemes:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec['scheme']} (Confidence: {rec['confidence']:.2f})")
        print()
    
    # Demo 4: Cross-Modal Understanding
    print("\n🔗 Demo 4: Cross-Modal Reasoning")
    multimodal_input = {
        "documents": {
            "text": "Community Forest Rights claim for 50 hectares in Telangana forest area"
        },
        "structured_data": {
            "latitude": [17.8],
            "longitude": [79.5],
            "indices": [0.9, 0.1, 0.95]  # Very high forest indicators
        },
        "task": "embedding"
    }
    
    with torch.no_grad():
        outputs = model(multimodal_input)
        embeddings = outputs['unified_embeddings']
        
        print("✅ Successfully fused:")
        print("  - Document text (CFR claim information)")
        print("  - Geospatial coordinates (Telangana location)")
        print("  - Satellite indices (High forest cover)")
        print(f"  - Into unified embedding: {embeddings.shape}")
        
        if 'attention_weights' in outputs and outputs['attention_weights'] is not None:
            print("✅ Cross-modal attention computed successfully")
            print("  - Model can reason about relationships between text and location")
    
    print("\n🎯 Demo Summary")
    print("=" * 30)
    print("✅ Unified model successfully demonstrates:")
    print("  1. Multimodal input processing (text + coordinates + indices)")
    print("  2. Natural language to SQL conversion")
    print("  3. Context-aware scheme recommendations")
    print("  4. Cross-modal attention and reasoning")
    print()
    print("🔑 Key Innovation: ONE model handles ALL tasks")
    print("   - No separate OCR → NER → GIS → LLM pipeline")
    print("   - Direct end-to-end multimodal reasoning")
    print("   - Shared embedding space for all modalities")
    print("   - Single API call for complex FRA analysis")
    
    print("\n🚀 Next Steps:")
    print("  1. Add training data: Place FRA documents in ./data/raw/")
    print("  2. Run data pipeline: python run.py --data-pipeline")
    print("  3. Train the model: python run.py --train")
    print("  4. Start API server: python run.py --serve")
    print("  5. Access WebGIS at: http://localhost:8000/")

def demo_api_capabilities():
    """Demo the API capabilities"""
    print("\n🌐 API Capabilities Preview")
    print("=" * 30)
    
    api_endpoints = {
        "Natural Language Queries": {
            "endpoint": "POST /query/natural-language",
            "example": '{"query": "Show all approved CFR claims in forest areas"}',
            "output": "SQL query + results"
        },
        "Document Processing": {
            "endpoint": "POST /document/process", 
            "example": "Upload FRA form image/PDF",
            "output": "Extracted entities + confidence scores"
        },
        "Satellite Analysis": {
            "endpoint": "POST /satellite/analyze",
            "example": '{"coordinates": [18.5, 79.0], "radius_km": 5}',
            "output": "Land use classification + spectral indices"
        },
        "DSS Recommendations": {
            "endpoint": "POST /dss/recommendations",
            "example": '{"village_name": "Sample", "coordinates": [18.5, 79.0]}',
            "output": "Ranked scheme recommendations"
        },
        "Claims Management": {
            "endpoint": "GET /claims/",
            "example": "?village=Adilabad&status=Approved",
            "output": "Filtered FRA claims list"
        }
    }
    
    for name, info in api_endpoints.items():
        print(f"\n{name}:")
        print(f"  Endpoint: {info['endpoint']}")
        print(f"  Example: {info['example']}")
        print(f"  Output: {info['output']}")
    
    print(f"\n💡 All endpoints powered by the SAME unified AI model!")

if __name__ == "__main__":
    try:
        demo_unified_model()
        demo_api_capabilities()
        
        print("\n" + "="*50)
        print("🎉 Demo completed successfully!")
        print("Ready to process real FRA data with unified AI!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("This is expected if dependencies are not installed.")
        print("Run: python run.py --setup  to install requirements.")
