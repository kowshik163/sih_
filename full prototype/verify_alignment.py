#!/usr/bin/env python3
"""
FRA AI Fusion System - Alignment Verification Script
Checks that all components are correctly aligned with the upgraded code
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all main components can be imported"""
    results = {}
    
    # Test config import
    try:
        from configs.config import config, validate_config
        results['config'] = '✅ SUCCESS'
    except Exception as e:
        results['config'] = f'❌ FAILED: {e}'
    
    # Test main model import
    try:
        from main_fusion_model import EnhancedFRAUnifiedEncoder, MultimodalPretrainingObjectives
        results['main_model'] = '✅ SUCCESS'
    except Exception as e:
        results['main_model'] = f'❌ FAILED: {e}'
    
    # Test data pipeline import
    try:
        sys.path.append(str(project_root / "1_data_processing"))
        from data_pipeline import EnhancedFRADataProcessor
        results['data_pipeline'] = '✅ SUCCESS'
    except Exception as e:
        results['data_pipeline'] = f'❌ FAILED: {e}'
    
    # Test training pipeline import
    try:
        sys.path.append(str(project_root / "2_model_fusion"))
        from train_fusion import EnhancedFRATrainingPipeline
        results['training_pipeline'] = '✅ SUCCESS'
    except Exception as e:
        results['training_pipeline'] = f'❌ FAILED: {e}'
    
    # Test API import
    try:
        sys.path.append(str(project_root / "3_webgis_backend"))
        from api import app
        results['api'] = '✅ SUCCESS'
    except Exception as e:
        results['api'] = f'❌ FAILED: {e}'
    
    return results

def test_model_interface():
    """Test that the enhanced model has the expected interface"""
    try:
        from main_fusion_model import EnhancedFRAUnifiedEncoder
        
        # Create a test config
        config = {
            'hidden_size': 1024,
            'unified_token_fusion': True,
            'temporal_modeling': {'enabled': False},
            'graph_neural_network': {'enabled': False},
            'memory_module': {'enabled': False}
        }
        
        # Try to instantiate the model
        model = EnhancedFRAUnifiedEncoder(config)
        
        # Check if expected methods exist
        methods = ['forward', 'generate_sql', 'recommend_schemes']
        missing_methods = []
        for method in methods:
            if not hasattr(model, method):
                missing_methods.append(method)
        
        if missing_methods:
            return f'❌ FAILED: Missing methods: {missing_methods}'
        else:
            return '✅ SUCCESS: All expected methods found'
            
    except Exception as e:
        return f'❌ FAILED: {e}'

def test_config_structure():
    """Test that config structure is compatible"""
    try:
        from configs.config import config
        
        # Check if config has the expected structure
        required_sections = ['model', 'training', 'data', 'api', 'database']
        missing_sections = []
        
        for section in required_sections:
            if not hasattr(config, section + '_config'):
                missing_sections.append(section)
        
        if missing_sections:
            return f'❌ FAILED: Missing config sections: {missing_sections}'
        
        # Check if config.config exists (for training pipeline)
        if not hasattr(config, 'config'):
            return '❌ FAILED: config.config attribute missing'
            
        return '✅ SUCCESS: Config structure is compatible'
        
    except Exception as e:
        return f'❌ FAILED: {e}'

def main():
    """Main verification function"""
    print("🔍 FRA AI Fusion System - Alignment Verification")
    print("=" * 60)
    
    print("\\n📦 Testing Component Imports:")
    import_results = test_imports()
    for component, result in import_results.items():
        print(f"  {component:<20}: {result}")
    
    print("\\n🧠 Testing Model Interface:")
    model_result = test_model_interface()
    print(f"  Enhanced Model      : {model_result}")
    
    print("\\n⚙️  Testing Config Structure:")
    config_result = test_config_structure()
    print(f"  Config Compatibility: {config_result}")
    
    print("\\n📁 Checking File Structure:")
    expected_files = [
        "1_data_processing/data_pipeline.py",
        "2_model_fusion/train_fusion.py",
        "3_webgis_backend/api.py",
        "main_fusion_model.py",
        "configs/config.py",
        "configs/config.json",
        "run.py"
    ]
    
    for file_path in expected_files:
        if (project_root / file_path).exists():
            print(f"  {file_path:<30}: ✅ EXISTS")
        else:
            print(f"  {file_path:<30}: ❌ MISSING")
    
    print("\\n🎯 Summary:")
    all_imports_successful = all('SUCCESS' in result for result in import_results.values())
    model_interface_ok = 'SUCCESS' in model_result
    config_structure_ok = 'SUCCESS' in config_result
    
    if all_imports_successful and model_interface_ok and config_structure_ok:
        print("  ✅ ALL CHECKS PASSED - System is properly aligned!")
        print("  🚀 Ready to run: python run.py --complete")
    else:
        print("  ⚠️  SOME ISSUES FOUND - Check details above")
        print("  📋 Install dependencies: pip install -r requirements.txt")
    
    print("\\n" + "=" * 60)

if __name__ == "__main__":
    main()
