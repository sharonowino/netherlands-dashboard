#!/usr/bin/env python3
"""Extract feature names from trained model"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

try:
    import joblib
    import pickle
    
    models_dir = Path(__file__).parent.parent / 'backend' / 'models'
    rf_path = models_dir / 'model_RandomForest.pkl'
    
    print(f"Loading model from: {rf_path}")
    print(f"File exists: {rf_path.exists()}")
    
    if not rf_path.exists():
        print("ERROR: Model file not found")
        sys.exit(1)
    
    # Try joblib first
    try:
        obj = joblib.load(rf_path)
        print(f"Loaded with joblib: {type(obj)}")
    except Exception as e:
        print(f"Joblib failed: {e}, trying pickle...")
        with open(rf_path, 'rb') as f:
            obj = pickle.load(f)
        print(f"Loaded with pickle: {type(obj)}")
    
    # Unwrap if dict
    if isinstance(obj, dict):
        print(f"Dictionary keys: {list(obj.keys())}")
        model = obj.get('model') or obj.get('trained_model') or obj
    else:
        model = obj
    
    print(f"\nModel type: {type(model).__name__}")
    print(f"n_features_in_: {getattr(model, 'n_features_in_', 'N/A')}")
    
    # Get feature names
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
        print(f"\nFound {len(features)} features:")
        print("\nFEATURE_NAMES: List[str] = [")
        for feat in features:
            print(f'    "{feat}",')
        print("]")
    else:
        print("\nNo feature_names_in_ attribute found")
        print(f"Available attributes: {[a for a in dir(model) if not a.startswith('_')][:10]}")
        
        # Check metadata for feature names
        if isinstance(obj, dict) and 'metadata' in obj:
            print(f"\nMetadata keys: {list(obj['metadata'].keys())}")
            
            # Print all metadata
            for key, value in obj['metadata'].items():
                if key not in ['model', 'trained_model'] and not callable(value):
                    val_str = str(value)[:200] if not isinstance(value, list) else f"list of {len(value)} items"
                    print(f"  {key}: {val_str}")
            
            if 'feature_names' in obj['metadata']:
                features = obj['metadata']['feature_names']
                print(f"\nFound {len(features)} features in metadata (model expects {getattr(model, 'n_features_in_', 'N/A')}):")
                print("\nFEATURE_NAMES: List[str] = [")
                for feat in features:
                    print(f'    "{feat}",')
                print("]")
                
                # If there's a mismatch, note it
                n_features = getattr(model, 'n_features_in_', None)
                if n_features and len(features) != n_features:
                    print(f"\n⚠️  WARNING: Metadata has {len(features)} features but model expects {n_features}")
                    print(f"   You may need to select the correct {n_features} features from the list above")
                    
            elif 'features' in obj['metadata']:
                features = obj['metadata']['features']
                print(f"\nFound {len(features)} features in metadata:")
                print("\nFEATURE_NAMES: List[str] = [")
                for feat in features:
                    print(f'    "{feat}",')
                print("]")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
