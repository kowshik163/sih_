"""
WebGIS Backend for FRA AI Fusion System
FastAPI backend with PostGIS integration and AI model serving
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import asyncio
import json
import os
import sys
import io
import psycopg2
import geopandas as gpd
from sqlalchemy import create_engine, text
import torch
from PIL import Image
import numpy as np
from datetime import datetime
import logging

# Add parent directory to path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_fusion_model import EnhancedFRAUnifiedEncoder

# Initialize FastAPI app
app = FastAPI(
    title="FRA AI Fusion API",
    description="Forest Rights Act monitoring system with unified AI capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
DB_ENGINE = None

# Pydantic models for API
class FRAQuery(BaseModel):
    query: str
    filters: Optional[Dict] = None

class DocumentUpload(BaseModel):
    file_name: str
    content_type: str

class SatelliteQuery(BaseModel):
    coordinates: Tuple[float, float]
    radius_km: float = 5.0
    date_range: Optional[Tuple[str, str]] = None

class DSSSuggestion(BaseModel):
    village_name: str
    coordinates: Tuple[float, float]
    population_data: Optional[Dict] = None

class FRAClaim(BaseModel):
    id: Optional[int] = None
    village_name: str
    patta_holder: str
    claim_type: str
    status: str
    coordinates: Tuple[float, float]
    area_hectares: float
    submission_date: str

# Database configuration
DATABASE_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "fra_gis"),
    "user": os.getenv("POSTGRES_USER", "fra_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "fra_password")
}

MODEL_CONFIG = {
    'hidden_size': 1024,
    'num_ner_labels': 10,
    'num_schemes': 50,
    'unified_token_fusion': True,
    'use_small_llm': True,
    'temporal_model': {'enabled': True, 'window': 12, 'd_model': 256},
    'geo_graph': {'enabled': True, 'k_neighbors': 8},
    'memory': {'enabled': True, 'capacity': 2048, 'eviction': 'lru'}
}

@app.on_event("startup")
async def startup_event():
    """Initialize model and database connections on startup"""
    global MODEL, DB_ENGINE
    
    try:
        # Initialize AI model
        MODEL = EnhancedFRAUnifiedEncoder(MODEL_CONFIG)
        
        # Load trained weights if available
        model_path = "../../2_model_fusion/checkpoints/final_model.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            MODEL.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded trained model weights")
        else:
            print("⚠️ Using untrained model weights")
        
        MODEL.eval()
        
        # Initialize database connection
        db_url = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
        DB_ENGINE = create_engine(db_url)
        
        # Test database connection
        with DB_ENGINE.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
        
        print("🚀 FRA AI Fusion API started successfully")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        # Continue without model/db for development
        MODEL = None
        DB_ENGINE = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FRA AI Fusion API",
        "version": "1.0.0",
        "model_loaded": MODEL is not None,
        "database_connected": DB_ENGINE is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with enhanced capabilities"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if MODEL else "not loaded",
        "database_status": "connected" if DB_ENGINE else "not connected",
        "temporal_enabled": MODEL_CONFIG.get('temporal_model', {}).get('enabled', False),
        "geo_graph_enabled": MODEL_CONFIG.get('geo_graph', {}).get('enabled', False),
        "memory_enabled": MODEL_CONFIG.get('memory', {}).get('enabled', False),
        "unified_token_fusion": MODEL_CONFIG.get('unified_token_fusion', False)
    }
# --- New API Endpoints for Advanced Multimodal Capabilities ---

@app.post("/temporal/analyze")
async def analyze_temporal_data(query: SatelliteQuery):
    """Analyze temporal sequences for a given location"""
    if not MODEL:
        raise HTTPException(status_code=503, detail="AI model not available")
    # Mock: In real implementation, fetch and process time series data
    lat, lon = query.coordinates
    result = {
        "coordinates": [lat, lon],
        "temporal_window": MODEL_CONFIG['temporal_model']['window'],
        "trend": "Increasing NDVI over last 12 months",
        "temporal_scores": [0.45, 0.48, 0.52, 0.55, 0.60, 0.62, 0.65, 0.67, 0.70, 0.72, 0.74, 0.76]
    }
    return result

@app.post("/geo/graph-query")
async def geo_graph_query(query: SatelliteQuery):
    """Query geospatial graph relationships for a location"""
    if not MODEL:
        raise HTTPException(status_code=503, detail="AI model not available")
    lat, lon = query.coordinates
    # Mock: In real implementation, run GNN over spatial graph
    result = {
        "coordinates": [lat, lon],
        "neighbors": [
            {"village": "Village A", "distance_km": 2.1},
            {"village": "Village B", "distance_km": 3.4}
        ],
        "graph_score": 0.82
    }
    return result

@app.post("/dss/advanced-recommendations")
async def advanced_dss_recommendations(suggestion: DSSSuggestion):
    """Get advanced DSS recommendations using knowledge graph and multi-objective optimization"""
    if not MODEL:
        raise HTTPException(status_code=503, detail="AI model not available")
    # Mock: In real implementation, use knowledge graph and optimization
    recommendations = [
        {"scheme": "PM-KISAN", "confidence": 0.95, "explanation": "High farmer population"},
        {"scheme": "Jal Jeevan Mission", "confidence": 0.88, "explanation": "Low water access"},
        {"scheme": "MGNREGA", "confidence": 0.80, "explanation": "High unemployment"}
    ]
    return {
        "village_name": suggestion.village_name,
        "recommendations": recommendations,
        "multi_objective": {"coverage": 0.92, "cost": 0.75, "impact": 0.89}
    }

@app.post("/diagnostics/multimodal-pretraining")
async def multimodal_pretraining_diagnostics():
    """Run diagnostics on multimodal pretraining objectives"""
    if not MODEL:
        raise HTTPException(status_code=503, detail="AI model not available")
    # Mock: In real implementation, return loss curves and metrics
    return {
        "contrastive_loss": 0.23,
        "masked_modeling_loss": 0.18,
        "cross_modal_accuracy": 0.91
    }

@app.post("/batch/process")
async def batch_process(requests: List[FRAQuery]):
    """Batch process multiple natural language queries"""
    if not MODEL:
        raise HTTPException(status_code=503, detail="AI model not available")
    results = []
    for query in requests:
        sql_query = MODEL.generate_sql(query.query)
        results.append({"query": query.query, "generated_sql": sql_query})
    return {"results": results}

@app.post("/query/natural-language")
async def natural_language_query(query: FRAQuery):
    """Process natural language queries about FRA data"""
    if not MODEL:
        raise HTTPException(status_code=503, detail="AI model not available")
    
    try:
        # Generate SQL query using the model
        sql_query = MODEL.generate_sql(query.query)
        
        # Execute query if database is available
        if DB_ENGINE:
            with DB_ENGINE.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                columns = result.keys()
                
                # Convert to list of dictionaries
                data = [dict(zip(columns, row)) for row in rows]
        else:
            # Mock response for development
            data = [
                {
                    "village_name": "Sample Village",
                    "status": "Approved",
                    "claim_type": "Individual Forest Rights"
                }
            ]
        
        return {
            "query": query.query,
            "generated_sql": sql_query,
            "results": data,
            "count": len(data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/document/process")
async def process_document(file: UploadFile = File(...)):
    """Process uploaded FRA document using OCR and NER"""
    if not MODEL:
        raise HTTPException(status_code=503, detail="AI model not available")
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Process document using the model
        inputs = {
            "documents": {"text": ""},  # Will be filled by OCR
            "task": "ner"
        }
        
        # Mock processing for now (implement actual OCR integration)
        processed_result = {
            "document_id": file.filename,
            "extracted_text": "Sample extracted text from document",
            "entities": {
                "village_name": "Extracted Village",
                "patta_holder": "John Doe",
                "claim_type": "Individual Forest Rights",
                "status": "Pending"
            },
            "confidence_score": 0.85,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return processed_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/satellite/analyze")
async def analyze_satellite_data(query: SatelliteQuery):
    """Analyze satellite imagery for a given location"""
    if not MODEL:
        raise HTTPException(status_code=503, detail="AI model not available")
    
    try:
        lat, lon = query.coordinates
        
        # Mock satellite analysis (implement actual satellite data integration)
        analysis_result = {
            "coordinates": [lat, lon],
            "analysis_date": datetime.now().isoformat(),
            "land_use_classification": {
                "forest": 45.2,
                "agriculture": 30.1,
                "water": 8.7,
                "built_up": 16.0
            },
            "spectral_indices": {
                "ndvi": 0.65,
                "ndwi": 0.23,
                "evi": 0.58
            },
            "detected_assets": {
                "water_bodies": 2,
                "forest_patches": 5,
                "agricultural_fields": 8
            },
            "recommendations": [
                "High forest cover indicates good conservation status",
                "Water bodies available for community use",
                "Agricultural potential identified"
            ]
        }
        
        return analysis_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Satellite analysis failed: {str(e)}")

@app.post("/dss/recommendations")
async def get_dss_recommendations(suggestion: DSSSuggestion):
    """Get Decision Support System recommendations for a village"""
    if not MODEL:
        raise HTTPException(status_code=503, detail="AI model not available")
    
    try:
        # Prepare village data for DSS analysis
        village_data = {
            "latitude": [suggestion.coordinates[0]],
            "longitude": [suggestion.coordinates[1]],
            "indices": [0.5, 0.3, 0.7],  # Mock indices
            "census_data": [1000, 600, 0.7] if suggestion.population_data else [1000, 600, 0.7]
        }
        
        # Get recommendations from model
        recommendations = MODEL.recommend_schemes(village_data)
        
        # Add contextual information
        enhanced_recommendations = []
        for rec in recommendations:
            enhanced_rec = {
                **rec,
                "description": get_scheme_description(rec["scheme"]),
                "implementation_steps": get_implementation_steps(rec["scheme"]),
                "expected_timeline": "3-6 months",
                "required_documents": get_required_documents(rec["scheme"])
            }
            enhanced_recommendations.append(enhanced_rec)
        
        return {
            "village_name": suggestion.village_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "recommendations": enhanced_recommendations,
            "priority_actions": enhanced_recommendations[:3]  # Top 3
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DSS recommendations failed: {str(e)}")

@app.get("/claims/")
async def get_fra_claims(
    village: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get FRA claims with optional filtering"""
    try:
        if DB_ENGINE:
            # Build query with filters
            query = "SELECT * FROM fra_claims WHERE 1=1"
            params = {}
            
            if village:
                query += " AND village_name ILIKE :village"
                params["village"] = f"%{village}%"
            
            if status:
                query += " AND status = :status"
                params["status"] = status
            
            query += " ORDER BY id LIMIT :limit OFFSET :offset"
            params["limit"] = limit
            params["offset"] = offset
            
            with DB_ENGINE.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.fetchall()
                columns = result.keys()
                
                claims = [dict(zip(columns, row)) for row in rows]
        else:
            # Mock data for development
            claims = [
                {
                    "id": 1,
                    "village_name": "Sample Village 1",
                    "patta_holder": "Ram Singh",
                    "claim_type": "Individual Forest Rights",
                    "status": "Approved",
                    "coordinates": "18.5, 79.0",
                    "area_hectares": 2.5
                },
                {
                    "id": 2,
                    "village_name": "Sample Village 2",
                    "patta_holder": "Sita Devi",
                    "claim_type": "Community Forest Rights",
                    "status": "Pending",
                    "coordinates": "18.6, 79.1",
                    "area_hectares": 15.0
                }
            ]
        
        return {
            "claims": claims,
            "total": len(claims),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching claims: {str(e)}")

@app.post("/claims/")
async def create_fra_claim(claim: FRAClaim):
    """Create a new FRA claim"""
    try:
        if DB_ENGINE:
            query = """
                INSERT INTO fra_claims 
                (village_name, patta_holder, claim_type, status, coordinates, area_hectares, submission_date)
                VALUES (:village_name, :patta_holder, :claim_type, :status, :coordinates, :area_hectares, :submission_date)
                RETURNING id
            """
            
            params = {
                "village_name": claim.village_name,
                "patta_holder": claim.patta_holder,
                "claim_type": claim.claim_type,
                "status": claim.status,
                "coordinates": f"{claim.coordinates[0]},{claim.coordinates[1]}",
                "area_hectares": claim.area_hectares,
                "submission_date": claim.submission_date
            }
            
            with DB_ENGINE.connect() as conn:
                result = conn.execute(text(query), params)
                claim_id = result.fetchone()[0]
                conn.commit()
        else:
            # Mock response for development
            claim_id = 999
        
        return {
            "message": "FRA claim created successfully",
            "claim_id": claim_id,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating claim: {str(e)}")

@app.get("/analytics/dashboard")
async def get_dashboard_analytics():
    """Get analytics data for dashboard"""
    try:
        if DB_ENGINE:
            with DB_ENGINE.connect() as conn:
                # Get basic statistics
                stats_query = """
                    SELECT 
                        status,
                        COUNT(*) as count,
                        SUM(area_hectares) as total_area
                    FROM fra_claims 
                    GROUP BY status
                """
                result = conn.execute(text(stats_query))
                status_stats = [dict(zip(result.keys(), row)) for row in result.fetchall()]
        else:
            # Mock data
            status_stats = [
                {"status": "Approved", "count": 450, "total_area": 2250.5},
                {"status": "Pending", "count": 120, "total_area": 600.2},
                {"status": "Rejected", "count": 30, "total_area": 150.0}
            ]
        
        return {
            "status_distribution": status_stats,
            "total_claims": sum(stat["count"] for stat in status_stats),
            "total_area_hectares": sum(stat["total_area"] for stat in status_stats),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")

# Helper functions
def get_scheme_description(scheme_name: str) -> str:
    """Get description for a scheme"""
    descriptions = {
        "PM-KISAN": "Direct income support to farmers",
        "Jal Jeevan Mission": "Providing tap water connections to rural households",
        "MGNREGA": "Employment guarantee scheme for rural areas",
        "DAJGUA": "Convergence of schemes across three ministries",
        "Pradhan Mantri Awas Yojana": "Housing scheme for rural areas"
    }
    return descriptions.get(scheme_name, "Government welfare scheme")

def get_implementation_steps(scheme_name: str) -> List[str]:
    """Get implementation steps for a scheme"""
    return [
        "Verify eligibility criteria",
        "Collect required documents",
        "Submit application through designated channels",
        "Follow up with local authorities",
        "Monitor implementation progress"
    ]

def get_required_documents(scheme_name: str) -> List[str]:
    """Get required documents for a scheme"""
    return [
        "Aadhaar card",
        "Bank account details",
        "FRA patta certificate",
        "Village verification letter",
        "Income certificate"
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
