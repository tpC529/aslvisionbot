"""
Simple tests for ASL Vision Bot modules.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from src import dataset_loader
        print("✓ dataset_loader imported")
    except Exception as e:
        print(f"✗ dataset_loader failed: {e}")
    
    try:
        from src import sequence_model
        print("✓ sequence_model imported")
    except Exception as e:
        print(f"✗ sequence_model failed: {e}")
    
    try:
        from src import asl_grammar
        print("✓ asl_grammar imported")
    except Exception as e:
        print(f"✗ asl_grammar failed: {e}")
    
    try:
        from src import preprocessing
        print("✓ preprocessing imported")
    except Exception as e:
        print(f"✗ preprocessing failed: {e}")
    
    try:
        from src import utils
        print("✓ utils imported")
    except Exception as e:
        print(f"✗ utils failed: {e}")


def test_asl_grammar():
    """Test ASL grammar module"""
    print("\nTesting ASL grammar...")
    
    from src.asl_grammar import ASLInterpreter
    
    interpreter = ASLInterpreter()
    
    # Test simple interpretation
    signs = ['HELLO']
    result = interpreter.interpret_signs(signs)
    print(f"  Signs: {signs}")
    print(f"  English: {result['english']}")
    print(f"  Expression: {result['expression_type']}")
    
    # Test question detection
    signs = ['WHAT', 'YOUR', 'NAME']
    result = interpreter.interpret_signs(signs)
    print(f"\n  Signs: {signs}")
    print(f"  English: {result['english']}")
    print(f"  Is Question: {result['is_question']}")


def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    from src.utils import load_config
    
    config = load_config('config.yaml')
    if config:
        print("✓ Configuration loaded")
        print(f"  Camera default index: {config.get('camera', {}).get('default_index', 'N/A')}")
        print(f"  LSTM units: {config.get('models', {}).get('lstm_units', 'N/A')}")
    else:
        print("✗ Configuration not loaded")


def test_sequence_model():
    """Test sequence model creation"""
    print("\nTesting sequence model...")
    
    try:
        from src.sequence_model import ASLSequenceModel
        
        model = ASLSequenceModel(
            input_shape=(30, 63),
            num_classes=10,
            lstm_units=64,
            use_attention=True
        )
        
        print("✓ Sequence model created")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Num classes: {model.num_classes}")
        
    except Exception as e:
        print(f"✗ Sequence model failed: {e}")


def main():
    """Run all tests"""
    print("="*60)
    print("ASL Vision Bot - Module Tests")
    print("="*60)
    
    test_imports()
    test_asl_grammar()
    test_config_loading()
    test_sequence_model()
    
    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
