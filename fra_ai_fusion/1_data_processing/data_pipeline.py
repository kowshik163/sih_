"""
Data Processing Pipeline for FRA AI Fusion System
Handles OCR, document processing, satellite imagery, and data standardization
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
from PIL import Image
import pytesseract
from transformers import LayoutLMv3Processor
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point, Polygon
import json
import sqlite3
from datetime import datetime
import logging

class FRADataProcessor:
    """Main data processing pipeline for FRA documents and satellite data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        
        # Initialize processors
        self.layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        
        # OCR configuration
        pytesseract.pytesseract.tesseract_cmd = config.get("tesseract_path", "tesseract")
        
        # Database setup
        self.db_path = config.get("db_path", "fra_data.db")
        self.init_database()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def init_database(self):
        """Initialize SQLite database for storing processed data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # FRA claims table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fra_claims (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT UNIQUE,
                    village_name TEXT,
                    patta_holder TEXT,
                    claim_type TEXT,
                    status TEXT,
                    coordinates TEXT,
                    area_hectares REAL,
                    submission_date TEXT,
                    processed_date TEXT,
                    confidence_score REAL,
                    raw_text TEXT,
                    structured_data TEXT
                )
            """)
            
            # Satellite data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS satellite_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tile_id TEXT UNIQUE,
                    center_lat REAL,
                    center_lon REAL,
                    acquisition_date TEXT,
                    satellite_source TEXT,
                    land_use_classification TEXT,
                    ndvi_value REAL,
                    ndwi_value REAL,
                    forest_cover_percentage REAL,
                    file_path TEXT
                )
            """)
            
            # Villages master data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS villages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    village_name TEXT,
                    district TEXT,
                    state TEXT,
                    latitude REAL,
                    longitude REAL,
                    population INTEGER,
                    tribal_population INTEGER,
                    literacy_rate REAL,
                    forest_area_hectares REAL
                )
            """)
            
            conn.commit()
            self.logger.info("Database initialized successfully")

class DocumentProcessor:
    """Process FRA documents using OCR and NER"""
    
    def __init__(self, parent: FRADataProcessor):
        self.parent = parent
        self.layout_processor = parent.layout_processor
        
    def process_fra_document(self, image_path: str) -> Dict:
        """
        Process a single FRA document image
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Dictionary with extracted information
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # OCR extraction
            ocr_result = self.extract_text_with_ocr(image)
            
            # Layout-aware processing
            layout_result = self.extract_structured_layout(image)
            
            # Named Entity Recognition
            entities = self.extract_entities(ocr_result['text'])
            
            # Combine results
            processed_data = {
                'document_id': os.path.basename(image_path),
                'raw_text': ocr_result['text'],
                'confidence_score': ocr_result['confidence'],
                'layout_data': layout_result,
                'entities': entities,
                'processed_date': datetime.now().isoformat()
            }
            
            # Store in database
            self.store_fra_claim(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.parent.logger.error(f"Error processing document {image_path}: {e}")
            return {}
    
    def extract_text_with_ocr(self, image: Image.Image) -> Dict:
        """Extract text using Tesseract OCR"""
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocessing for better OCR
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # OCR with confidence scores
        data = pytesseract.image_to_data(denoised, output_type=pytesseract.Output.DICT)
        
        # Extract text and calculate average confidence
        text_parts = []
        confidences = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Filter low confidence
                text_parts.append(data['text'][i])
                confidences.append(int(data['conf'][i]))
        
        full_text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'text': full_text,
            'confidence': avg_confidence,
            'word_data': data
        }
    
    def extract_structured_layout(self, image: Image.Image) -> Dict:
        """Extract structured layout information using LayoutLMv3"""
        try:
            # Process with LayoutLMv3
            encoding = self.layout_processor(image, return_tensors="pt")
            
            return {
                'input_ids': encoding['input_ids'].tolist(),
                'bbox': encoding.get('bbox', []).tolist(),
                'pixel_values': encoding.get('pixel_values', []).shape if 'pixel_values' in encoding else None
            }
        except Exception as e:
            self.parent.logger.warning(f"Layout processing failed: {e}")
            return {}
    
    def extract_entities(self, text: str) -> Dict:
        """Extract named entities from text"""
        entities = {
            'village_name': '',
            'patta_holder': '',
            'claim_type': '',
            'status': '',
            'coordinates': '',
            'area': ''
        }
        
        # Simple rule-based NER (can be replaced with trained model)
        import re
        
        # Village name pattern
        village_match = re.search(r'Village[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if village_match:
            entities['village_name'] = village_match.group(1).strip()
        
        # Patta holder pattern
        patta_match = re.search(r'(?:Patta\s+Holder|Name)[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if patta_match:
            entities['patta_holder'] = patta_match.group(1).strip()
        
        # Claim type
        if 'IFR' in text.upper():
            entities['claim_type'] = 'Individual Forest Rights'
        elif 'CFR' in text.upper():
            entities['claim_type'] = 'Community Forest Rights'
        elif 'CR' in text.upper():
            entities['claim_type'] = 'Community Rights'
        
        # Status
        if any(word in text.upper() for word in ['APPROVED', 'GRANTED']):
            entities['status'] = 'Approved'
        elif any(word in text.upper() for word in ['REJECTED', 'DENIED']):
            entities['status'] = 'Rejected'
        elif 'PENDING' in text.upper():
            entities['status'] = 'Pending'
        
        # Coordinates (lat, lon pattern)
        coord_match = re.search(r'(\d+\.?\d*)[,\s]+(\d+\.?\d*)', text)
        if coord_match:
            entities['coordinates'] = f"{coord_match.group(1)},{coord_match.group(2)}"
        
        return entities
    
    def store_fra_claim(self, data: Dict):
        """Store processed FRA claim in database"""
        entities = data.get('entities', {})
        
        with sqlite3.connect(self.parent.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO fra_claims 
                (document_id, village_name, patta_holder, claim_type, status, 
                 coordinates, confidence_score, raw_text, structured_data, processed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['document_id'],
                entities.get('village_name', ''),
                entities.get('patta_holder', ''),
                entities.get('claim_type', ''),
                entities.get('status', ''),
                entities.get('coordinates', ''),
                data.get('confidence_score', 0),
                data.get('raw_text', ''),
                json.dumps(data),
                data['processed_date']
            ))
            
            conn.commit()


class SatelliteProcessor:
    """Process satellite imagery for land use classification and asset mapping"""
    
    def __init__(self, parent: FRADataProcessor):
        self.parent = parent
        
    def process_satellite_tile(self, raster_path: str, metadata: Dict = None) -> Dict:
        """
        Process a satellite image tile
        
        Args:
            raster_path: Path to satellite raster file (GeoTIFF)
            metadata: Additional metadata about the tile
            
        Returns:
            Dictionary with processed results
        """
        try:
            with rasterio.open(raster_path) as src:
                # Read bands (assuming Sentinel-2 or similar)
                bands = src.read()  # Shape: (bands, height, width)
                transform = src.transform
                crs = src.crs
                
                # Calculate center coordinates
                height, width = bands.shape[1], bands.shape[2]
                center_x, center_y = transform * (width // 2, height // 2)
                
                # Convert to lat/lon if needed
                if crs.to_string() != 'EPSG:4326':
                    import pyproj
                    transformer = pyproj.Transformer.from_crs(crs, 'EPSG:4326')
                    center_lat, center_lon = transformer.transform(center_y, center_x)
                else:
                    center_lat, center_lon = center_y, center_x
                
                # Calculate spectral indices
                indices = self.calculate_spectral_indices(bands)
                
                # Land use classification
                land_use_map = self.classify_land_use(bands)
                
                # Asset detection
                assets = self.detect_assets(bands, land_use_map)
                
                # Store results
                tile_data = {
                    'tile_id': os.path.basename(raster_path),
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'acquisition_date': metadata.get('date', ''),
                    'satellite_source': metadata.get('source', 'Unknown'),
                    'spectral_indices': indices,
                    'land_use_classification': land_use_map.tolist(),
                    'detected_assets': assets,
                    'file_path': raster_path
                }
                
                self.store_satellite_data(tile_data)
                
                return tile_data
                
        except Exception as e:
            self.parent.logger.error(f"Error processing satellite tile {raster_path}: {e}")
            return {}
    
    def calculate_spectral_indices(self, bands: np.ndarray) -> Dict:
        """Calculate common spectral indices"""
        # Assuming Sentinel-2 band order: [B2, B3, B4, B8, B11, B12]
        # B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2
        
        indices = {}
        
        if bands.shape[0] >= 4:  # Need at least 4 bands
            blue = bands[0].astype(float)
            green = bands[1].astype(float)
            red = bands[2].astype(float)
            nir = bands[3].astype(float)
            
            # NDVI (Normalized Difference Vegetation Index)
            ndvi = (nir - red) / (nir + red + 1e-8)
            indices['ndvi'] = float(np.nanmean(ndvi))
            
            # NDWI (Normalized Difference Water Index)
            if bands.shape[0] >= 5:
                swir1 = bands[4].astype(float)
                ndwi = (green - swir1) / (green + swir1 + 1e-8)
                indices['ndwi'] = float(np.nanmean(ndwi))
            
            # EVI (Enhanced Vegetation Index)
            evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
            indices['evi'] = float(np.nanmean(evi))
            
        return indices
    
    def classify_land_use(self, bands: np.ndarray) -> np.ndarray:
        """Classify land use using simple thresholding (can be replaced with ML model)"""
        # Simplified classification based on spectral indices
        height, width = bands.shape[1], bands.shape[2]
        classification = np.zeros((height, width), dtype=np.uint8)
        
        if bands.shape[0] >= 4:
            red = bands[2].astype(float)
            nir = bands[3].astype(float)
            
            # Calculate NDVI
            ndvi = (nir - red) / (nir + red + 1e-8)
            
            # Simple thresholding
            # 0: Background/Other
            # 1: Forest (high NDVI)
            # 2: Agriculture (medium NDVI)
            # 3: Water (low NDVI, high blue)
            # 4: Built-up (low NDVI)
            
            forest_mask = ndvi > 0.6
            agriculture_mask = (ndvi > 0.3) & (ndvi <= 0.6)
            water_mask = (ndvi < 0) & (bands[0] > bands[2])  # Blue > Red
            
            classification[forest_mask] = 1
            classification[agriculture_mask] = 2
            classification[water_mask] = 3
            # Everything else remains 0 (other/built-up)
        
        return classification
    
    def detect_assets(self, bands: np.ndarray, land_use_map: np.ndarray) -> Dict:
        """Detect key assets like water bodies, homesteads, etc."""
        assets = {
            'water_bodies': 0,
            'forest_patches': 0,
            'agricultural_fields': 0,
            'built_up_areas': 0
        }
        
        # Count pixels for each land use type
        unique, counts = np.unique(land_use_map, return_counts=True)
        pixel_area = 100  # Assuming 10m resolution (100 m² per pixel)
        
        for land_use_id, pixel_count in zip(unique, counts):
            area_hectares = (pixel_count * pixel_area) / 10000  # Convert to hectares
            
            if land_use_id == 1:  # Forest
                assets['forest_patches'] = area_hectares
            elif land_use_id == 2:  # Agriculture
                assets['agricultural_fields'] = area_hectares
            elif land_use_id == 3:  # Water
                assets['water_bodies'] = area_hectares
            elif land_use_id == 4:  # Built-up
                assets['built_up_areas'] = area_hectares
        
        return assets
    
    def store_satellite_data(self, data: Dict):
        """Store satellite processing results in database"""
        with sqlite3.connect(self.parent.db_path) as conn:
            cursor = conn.cursor()
            
            indices = data.get('spectral_indices', {})
            
            cursor.execute("""
                INSERT OR REPLACE INTO satellite_data 
                (tile_id, center_lat, center_lon, acquisition_date, satellite_source,
                 land_use_classification, ndvi_value, ndwi_value, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['tile_id'],
                data['center_lat'],
                data['center_lon'],
                data['acquisition_date'],
                data['satellite_source'],
                json.dumps(data.get('detected_assets', {})),
                indices.get('ndvi', 0),
                indices.get('ndwi', 0),
                data['file_path']
            ))
            
            conn.commit()


class DataIntegrator:
    """Integrate processed data and prepare for model training"""
    
    def __init__(self, parent: FRADataProcessor):
        self.parent = parent
        
    def create_training_pairs(self) -> List[Dict]:
        """Create paired training data for multimodal learning"""
        training_pairs = []
        
        with sqlite3.connect(self.parent.db_path) as conn:
            # Query FRA claims with coordinates
            query = """
                SELECT f.*, v.latitude, v.longitude, v.district, v.state
                FROM fra_claims f
                LEFT JOIN villages v ON f.village_name = v.village_name
                WHERE f.coordinates IS NOT NULL AND f.coordinates != ''
            """
            
            claims_df = pd.read_sql_query(query, conn)
            
            for _, claim in claims_df.iterrows():
                # Find corresponding satellite data
                if claim['coordinates']:
                    try:
                        lat, lon = map(float, claim['coordinates'].split(','))
                        
                        # Find nearest satellite tile
                        sat_query = """
                            SELECT * FROM satellite_data
                            WHERE ABS(center_lat - ?) < 0.01 AND ABS(center_lon - ?) < 0.01
                            ORDER BY ABS(center_lat - ?) + ABS(center_lon - ?) ASC
                            LIMIT 1
                        """
                        
                        sat_data = pd.read_sql_query(sat_query, conn, params=[lat, lon, lat, lon])
                        
                        if not sat_data.empty:
                            training_pair = {
                                'document_data': {
                                    'text': claim['raw_text'],
                                    'entities': json.loads(claim['structured_data']) if claim['structured_data'] else {}
                                },
                                'satellite_data': {
                                    'tile_path': sat_data.iloc[0]['file_path'],
                                    'ndvi': sat_data.iloc[0]['ndvi_value'],
                                    'ndwi': sat_data.iloc[0]['ndwi_value'],
                                    'land_use': json.loads(sat_data.iloc[0]['land_use_classification']) if sat_data.iloc[0]['land_use_classification'] else {}
                                },
                                'coordinates': [lat, lon],
                                'labels': {
                                    'village': claim['village_name'],
                                    'status': claim['status'],
                                    'claim_type': claim['claim_type']
                                }
                            }
                            
                            training_pairs.append(training_pair)
                            
                    except (ValueError, IndexError) as e:
                        self.parent.logger.warning(f"Error parsing coordinates: {claim['coordinates']}")
        
        self.parent.logger.info(f"Created {len(training_pairs)} training pairs")
        return training_pairs
    
    def export_training_data(self, output_path: str):
        """Export training data in format suitable for model training"""
        training_pairs = self.create_training_pairs()
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(training_pairs, f, indent=2)
        
        self.parent.logger.info(f"Training data exported to {output_path}")
        return training_pairs


# Main processing pipeline
def main():
    """Main data processing pipeline"""
    config = {
        "tesseract_path": "tesseract",  # Update with actual path
        "db_path": "fra_data.db"
    }
    
    # Initialize processor
    processor = FRADataProcessor(config)
    doc_processor = DocumentProcessor(processor)
    sat_processor = SatelliteProcessor(processor)
    integrator = DataIntegrator(processor)
    
    # Example usage
    print("FRA Data Processing Pipeline")
    print("1. Process FRA documents")
    print("2. Process satellite tiles")
    print("3. Create training pairs")
    
    # Process sample documents
    documents_dir = "sample_documents/"
    if os.path.exists(documents_dir):
        for doc_file in os.listdir(documents_dir):
            if doc_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                doc_path = os.path.join(documents_dir, doc_file)
                result = doc_processor.process_fra_document(doc_path)
                print(f"Processed document: {doc_file}")
    
    # Export training data
    training_data = integrator.export_training_data("training_data.json")
    print(f"Exported {len(training_data)} training pairs")

if __name__ == "__main__":
    main()
