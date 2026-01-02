"""
ASL Vision Bot - Source Package

This package contains modules for ASL recognition:
- dataset_loader: Load and preprocess datasets
- sequence_model: LSTM/Attention models for sequence recognition
- asl_grammar: ASL grammar rules and interpretation
- preprocessing: Video and image preprocessing
- utils: Utility functions
"""

__version__ = '2.0.0'
__author__ = 'ASL Vision Bot Contributors'

from .dataset_loader import DatasetLoader, KaggleDatasetLoader, How2SignDatasetLoader
from .sequence_model import ASLSequenceModel, TemporalSequenceProcessor
from .asl_grammar import ASLInterpreter, ASLGrammarRules, ContextManager
from .preprocessing import VideoPreprocessor, DataAugmentation
from .utils import load_config, setup_logging

__all__ = [
    'DatasetLoader',
    'KaggleDatasetLoader',
    'How2SignDatasetLoader',
    'ASLSequenceModel',
    'TemporalSequenceProcessor',
    'ASLInterpreter',
    'ASLGrammarRules',
    'ContextManager',
    'VideoPreprocessor',
    'DataAugmentation',
    'load_config',
    'setup_logging',
]
