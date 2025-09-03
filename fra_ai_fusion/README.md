# 🌲 FRA AI Fusion System

## Forest Rights Act Monitoring with Unified AI Architecture

A comprehensive AI system that combines OCR, Computer Vision, NLP, and Geospatial capabilities into a single unified model for Forest Rights Act (FRA) monitoring and decision support.

---

## 🎯 Project Overview

The FRA AI Fusion System addresses the challenges in Forest Rights Act implementation by creating a **unified AI model** that inherently understands:

- 📄 **Document Processing**: OCR and NER for FRA forms and legal documents
- 🛰️ **Satellite Analysis**: Land use classification and asset mapping
- 🗺️ **Geospatial Intelligence**: Spatial relationships and geographic queries
- 🧠 **Decision Support**: Policy recommendations and scheme suggestions
- 📊 **Data Integration**: Seamless fusion of multimodal information

### Key Innovation
Instead of orchestrating separate AI tools, we **fuse capabilities at the model level**, creating a single AI that natively understands all modalities.

---

## 🏗️ Architecture

```
📁 fra_ai_fusion/
├── 🧠 main_fusion_model.py          # Core unified AI model
├── 📊 1_data_processing/            # Data ingestion & preprocessing
│   └── data_pipeline.py            # OCR, satellite processing, NER
├── 🔧 2_model_fusion/               # Model training & fusion
│   └── train_fusion.py             # Multi-stage curriculum training
├── 🌐 3_webgis_backend/             # API server & PostGIS integration
│   └── api.py                      # FastAPI backend with AI endpoints
├── 🎨 4_frontend/                   # React-based FRA Atlas (TODO)
├── 💡 5_dss_engine/                 # Decision Support System (TODO)
├── 🚀 6_deployment/                 # Docker & Kubernetes configs (TODO)
├── 📈 7_evaluation/                 # Testing & monitoring (TODO)
├── ⚙️ configs/                      # Configuration management
│   └── config.py                   # Centralized config
├── 📋 requirements.txt              # Dependencies
└── 🚀 run.py                       # Main orchestration script
```

---

## ✨ Key Features

### 🧠 Unified AI Model
- **Multimodal Encoder**: Projects text, images, and structured data to shared embedding space
- **Cross-Modal Attention**: Enables reasoning across different data types
- **Task-Specific Heads**: NER, segmentation, SQL generation, recommendations
- **Geospatially-Aware**: Understanding of spatial relationships and coordinates

### 📊 Comprehensive Data Processing
- **OCR Pipeline**: Tesseract + LayoutLMv3 for structured document extraction
- **Satellite Analysis**: Land use classification with DeepLabV3+ and spectral indices
- **NER Extraction**: Village names, patta holders, claim status, coordinates
- **Data Integration**: Automatic pairing of documents with satellite imagery

### 🌐 WebGIS Integration
- **FastAPI Backend**: RESTful API with AI-powered endpoints
- **PostGIS Database**: Spatial data storage and querying
- **Natural Language Queries**: Convert English to SQL automatically
- **Real-time Processing**: Upload documents and get instant AI analysis

### 💡 Decision Support System
- **Scheme Recommendations**: AI-powered suggestions for CSS schemes
- **Policy Analysis**: Correlate FRA data with development programs
- **Priority Ranking**: Intelligent prioritization of interventions
- **Contextual Insights**: Location and community-specific recommendations

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- PostgreSQL with PostGIS extension
- 16GB+ RAM (for full model)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd fra_ai_fusion
```

2. **Setup environment**
```bash
python run.py --setup
```

3. **Check system status**
```bash
python run.py --status
```

### Usage Options

#### 🔄 Complete Pipeline (Recommended)
```bash
# Run everything from data processing to API server
python run.py --complete --data-dir /path/to/your/fra/documents
```

#### 📋 Step-by-Step Execution

1. **Process your FRA data**
```bash
python run.py --data-pipeline --data-dir /path/to/documents
```

2. **Train the unified AI model**
```bash
python run.py --train
```

3. **Start the API server**
```bash
python run.py --serve --host 0.0.0.0 --port 8000
```

4. **Evaluate model performance**
```bash
python run.py --eval
```

---

## 🔧 Configuration

The system uses a centralized configuration in `configs/config.py`. Key settings:

```python
# Model Configuration
"model": {
    "hidden_size": 1024,
    "num_ner_labels": 15,
    "temperature": 0.7
}

# Training Configuration  
"training": {
    "learning_rate": 1e-4,
    "batch_size": 4,
    "num_epochs": {"stage_1": 10, "stage_2": 8}
}

# Database Configuration
"database": {
    "host": "localhost",
    "port": 5432,
    "name": "fra_gis"
}
```

---

## 🧠 Model Training Stages

The system implements **4-stage curriculum learning** as outlined in the technical documentation:

### Stage 1: Foundation Training (10 epochs)
- OCR correction and text extraction
- Named Entity Recognition for FRA fields
- Satellite image segmentation

### Stage 2: Cross-Modal Alignment (8 epochs)
- Contrastive learning between text and satellite data
- Unified embedding space creation
- Spatial relationship learning

### Stage 3: Tool Skills Training (5 epochs)
- SQL query generation from natural language
- API call formatting and execution
- Structured output generation

### Stage 4: Decision Support Training (5 epochs)
- Policy recommendation learning
- Scheme eligibility mapping
- Priority scoring and ranking

---

## 📡 API Endpoints

Once the server is running (`python run.py --serve`), available endpoints:

### Core AI Endpoints
- `POST /query/natural-language` - Convert English queries to SQL and execute
- `POST /document/process` - Upload and process FRA documents with OCR+NER
- `POST /satellite/analyze` - Analyze satellite imagery for land use and assets
- `POST /dss/recommendations` - Get AI-powered scheme recommendations

### Data Management
- `GET /claims/` - Retrieve FRA claims with filtering
- `POST /claims/` - Create new FRA claims
- `GET /analytics/dashboard` - Get analytics for dashboard visualization

### Example Usage
```bash
# Natural language query
curl -X POST "http://localhost:8000/query/natural-language" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show all approved FRA claims in Telangana"}'

# Process document
curl -X POST "http://localhost:8000/document/process" \
  -F "file=@fra_document.pdf"

# Get DSS recommendations
curl -X POST "http://localhost:8000/dss/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"village_name": "Sample Village", "coordinates": [18.5, 79.0]}'
```

---

## 🎯 Use Cases

### For Government Officials
- **Document Digitization**: Convert paper FRA forms to structured digital data
- **Progress Monitoring**: Track FRA implementation across states/districts
- **Policy Planning**: Get AI recommendations for scheme implementation
- **Spatial Analysis**: Visualize claims overlaid with satellite imagery

### For Researchers
- **Data Analysis**: Query FRA database using natural language
- **Trend Identification**: Analyze approval patterns and success rates
- **Impact Assessment**: Correlate FRA rights with development outcomes

### For NGOs and Community Organizations
- **Claim Support**: Help communities prepare and track FRA applications
- **Resource Mapping**: Identify available assets and resources in FRA areas
- **Advocacy**: Use data insights to support community rights

---

## 📊 Expected Performance

Based on the technical architecture and training approach:

- **Document Processing**: 95%+ accuracy on FRA form digitization
- **Satellite Analysis**: 85%+ accuracy on land use classification
- **Query Processing**: Sub-second response for most spatial queries
- **Recommendation Quality**: Context-aware suggestions with 80%+ relevance

---

## 🛠️ Development & Extension

### Adding New Data Sources
1. Extend `DataProcessor` in `1_data_processing/data_pipeline.py`
2. Add new modality encoder in `main_fusion_model.py`
3. Update training pipeline in `2_model_fusion/train_fusion.py`

### Adding New Tasks
1. Create task-specific head in `FRAUnifiedEncoder`
2. Add training loss function
3. Create API endpoint in `3_webgis_backend/api.py`

### Custom Deployment
1. Modify `6_deployment/` configurations
2. Update database settings in `configs/config.py`
3. Scale using Kubernetes manifests

---

## 🚨 Important Notes

### Data Privacy & Security
- FRA data contains sensitive personal and community information
- Implement proper access controls and audit trails
- Use encryption for data at rest and in transit
- Follow government data protection guidelines

### Model Limitations
- Current implementation uses mock data for demonstration
- Full training requires substantial computational resources
- Production deployment needs security hardening
- Model performance depends on quality of training data

### Prerequisites for Production
- High-quality annotated FRA documents (1000+ samples)
- Satellite imagery with ground truth labels
- Proper database setup with PostGIS
- GPU infrastructure for training and inference

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation  
5. Submit a pull request

---

## 📄 License

This project is developed for Smart India Hackathon 2025 under the problem statement for Forest Rights Act monitoring.

---

## 🙏 Acknowledgments

- **Ministry of Tribal Affairs** for the problem statement
- **Smart India Hackathon** for the platform
- **Open source community** for the foundational tools and libraries

---

## 📞 Support

For questions and support:
- Check the system status: `python run.py --status`
- Review logs in `logs/` directory
- Validate configuration: `python -c "from configs.config import validate_config; print(validate_config())"`

---

**🌲 Building a unified AI for Forest Rights - One model, all capabilities! 🌲**
