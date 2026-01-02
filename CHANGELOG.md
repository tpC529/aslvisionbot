# Changelog

All notable changes to the ASL Vision Bot project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-01-02

### Added

#### Core Features
- **Sentence-Level Recognition**: System can now recognize complete ASL sentences, not just individual letters
- **ASL Grammar Support**: Proper ASL grammar rules for translation to English
- **Context-Aware Conversations**: Multi-turn conversation tracking and context management
- **Enhanced Mode**: Automatic detection of enhanced capabilities with graceful fallback

#### Infrastructure
- Comprehensive configuration system via `config.yaml`
- Modular architecture with separate source modules in `src/`
- Dataset loader utilities for Kaggle and How2Sign datasets
- Model training pipeline with LSTM/Attention architecture
- Data preprocessing and augmentation utilities

#### Documentation
- Complete README with installation and usage instructions
- Dataset attribution and licensing information (DATASETS.md)
- Model architecture documentation (MODEL_ARCHITECTURE.md)
- Training guide (TRAINING.md)
- Usage examples and API documentation (USAGE.md)
- LICENSE file with proper attributions

#### Models
- LSTM-based sequence model architecture
- Attention mechanism for focusing on relevant frames
- Temporal sequence processing
- Sliding window approach for continuous recognition

#### ASL Grammar & NLP
- ASL grammar rules implementation
- Topic-comment sentence structure support
- Time-first ordering
- Question detection (WH-questions and yes/no)
- Common phrase recognition
- Phrase prediction and completion
- Expression type detection (greeting, gratitude, etc.)

#### Scripts & Tools
- Dataset download script (`scripts/download_datasets.py`)
- Model download script (`scripts/download_models.py`)
- Model training script (`scripts/train_model.py`)
- Module testing script (`tests/test_modules.py`)

#### UI Enhancements
- Mode indicator in window title (Basic/Enhanced)
- Enhanced mode welcome message
- Updated instructions panel
- Better status messages

### Changed

#### Core Application (chat.py)
- `ASLRecognizer` class now supports configuration
- Added sign sequence tracking for sentence-level analysis
- Enhanced text interpretation using ASL grammar
- Improved gesture buffering and temporal smoothing

#### LLM Handler
- Context-aware response generation
- Conversation history tracking
- Better handling of ASL-translated inputs
- Configurable timeout and parameters

#### Camera Thread
- Configuration support
- Better frame processing

### Technical Details

#### Backward Compatibility
- Application works in Basic mode without enhanced dependencies
- Graceful degradation when modules unavailable
- All existing functionality preserved

#### Performance
- Configurable buffer sizes
- Adjustable confidence thresholds
- Caching support in configuration
- Optimized frame processing

#### Code Quality
- Comprehensive logging support
- Type hints in new modules
- Modular design
- Well-documented functions
- Consistent code style

### Dependencies
- Added: numpy, tensorflow, keras, pandas, scikit-learn, PyYAML, tqdm, matplotlib
- Added: Optional dependencies for dataset downloading (kaggle, gdown)
- Maintained: Existing dependencies (PyQt6, opencv-python, mediapipe, requests)

### File Structure
```
aslvisionbot/
├── chat.py                     [MODIFIED] Enhanced with sentence-level support
├── config.yaml                 [NEW] Configuration file
├── requirements.txt            [NEW] Python dependencies
├── README.md                   [NEW] Comprehensive documentation
├── LICENSE                     [NEW] MIT license with attributions
├── .gitignore                  [NEW] Git ignore rules
├── src/                        [NEW] Source modules
│   ├── __init__.py
│   ├── dataset_loader.py
│   ├── sequence_model.py
│   ├── asl_grammar.py
│   ├── preprocessing.py
│   └── utils.py
├── scripts/                    [NEW] Utility scripts
│   ├── download_datasets.py
│   ├── download_models.py
│   └── train_model.py
├── tests/                      [NEW] Test modules
│   └── test_modules.py
├── docs/                       [NEW] Documentation
│   ├── DATASETS.md
│   ├── MODEL_ARCHITECTURE.md
│   ├── TRAINING.md
│   └── USAGE.md
├── models/                     [NEW] Model directory
│   └── README.md
└── datasets/                   [NEW] Dataset directory
    └── README.md
```

## [1.0.0] - 2024 (Original Release)

### Initial Release

#### Features
- Basic hand tracking using MediaPipe
- Simple gesture recognition (A, B, L, O, V)
- Control gestures (SPACE, SUBMIT)
- LLM integration with Ollama
- Web search capability
- PyQt6 GUI interface
- Real-time video processing
- Chat interface with message history

#### Core Files
- `chat.py` - Main application
- `hand_landmarker.task` - MediaPipe model
- `asl_model.h5` - Basic gesture model
- `actions.npy` - Action labels

---

## Migration Guide (1.0.0 → 2.0.0)

### For Users

#### Minimal Changes
If you just want to use the basic features, no changes needed:
```bash
python chat.py  # Works as before in Basic mode
```

#### Enhanced Features
To use new sentence-level recognition:
```bash
pip install -r requirements.txt
python chat.py  # Automatically runs in Enhanced mode
```

### For Developers

#### Configuration
- Create `config.yaml` for custom settings
- Use `src.utils.load_config()` to load configuration

#### ASL Grammar
```python
from src.asl_grammar import ASLInterpreter

interpreter = ASLInterpreter()
result = interpreter.interpret_signs(['HELLO', 'MY', 'NAME'])
```

#### Dataset Loading
```python
from src.dataset_loader import DatasetLoader

loader = DatasetLoader(config)
images, labels, classes = loader.load_alphabet_dataset('kaggle')
```

#### Model Training
```bash
python scripts/download_datasets.py
python scripts/train_model.py
```

### Breaking Changes
None - fully backward compatible!

---

## Future Roadmap

### Version 2.1.0 (Planned)
- [ ] Two-handed sign recognition
- [ ] Facial expression integration
- [ ] Body pose support
- [ ] Real-time translation mode
- [ ] Mobile app version

### Version 2.2.0 (Planned)
- [ ] Multi-signer support
- [ ] Video recording and playback
- [ ] Custom sign creation
- [ ] Sign dictionary browser
- [ ] Performance optimizations

### Version 3.0.0 (Planned)
- [ ] Transformer-based models
- [ ] Multi-language support
- [ ] Cloud deployment
- [ ] API service
- [ ] Community features

---

## Acknowledgments

- MediaPipe team for hand tracking
- Kaggle community for ASL datasets
- How2Sign research team
- Ollama for local LLM
- All contributors and testers

---

For detailed information about any version, see the corresponding documentation in the `docs/` directory.
