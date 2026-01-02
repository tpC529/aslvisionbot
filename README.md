# ASL Vision Bot

An advanced American Sign Language (ASL) recognition chatbot that can understand and respond to complete sentences and phrases in sign language, powered by deep learning and natural language processing.

## 🌟 Features

### Current Capabilities
- **Real-time Hand Tracking**: Uses MediaPipe for accurate hand landmark detection
- **Basic Gesture Recognition**: Recognizes individual ASL letters (A, B, L, O, V) and control gestures
- **Interactive Chat Interface**: PyQt6-based GUI for seamless interaction
- **AI-Powered Responses**: Integration with Ollama LLM for intelligent conversation
- **Web Search Integration**: Can search the web for current information when needed

### Enhanced Capabilities (In Progress)
- **Sentence-Level Recognition**: Recognizes complete ASL sentences and phrases, not just individual signs
- **Temporal Sequence Modeling**: Understands the flow and context of continuous sign sequences
- **ASL Grammar Support**: Proper handling of ASL grammar rules (which differ from English)
- **Contextual Understanding**: NLP-powered interpretation of signed content
- **Multi-Turn Conversations**: Maintains context across conversation turns
- **Dataset Integration**: Leverages pre-trained models from Kaggle and How2Sign datasets

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Webcam/Camera device
- 4GB+ RAM recommended
- GPU recommended for model training (optional for inference)

### Software Dependencies
- OpenCV (cv2)
- NumPy
- MediaPipe
- PyQt6
- TensorFlow/Keras (for deep learning models)
- Requests (for API calls)
- Ollama (for LLM responses)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/tpC529/aslvisionbot.git
cd aslvisionbot
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install and Start Ollama
```bash
# Install Ollama (visit https://ollama.ai for platform-specific instructions)

# Start Ollama server
ollama serve

# In another terminal, pull the model
ollama pull llama3.2:1b
```

### 4. Download Pre-trained Models (Optional)
For sentence-level recognition, download the required models:
```bash
# Download from releases or train your own
python scripts/download_models.py
```

## 🎮 Usage

### Basic Usage
```bash
python chat.py
```

### Sign Language Input
1. **Individual Letters**: Hold hand gestures for 1 second
2. **Space**: Open palm facing down
3. **Submit**: Thumbs up gesture
4. **Auto-Submit**: 3 seconds of no gestures will auto-submit

### Supported Gestures
- **A**: Closed fist with thumb alongside
- **B**: Open palm with fingers together
- **L**: Index finger and thumb extended (L-shape)
- **O**: Thumb and index finger forming circle
- **V**: Index and middle fingers extended (V-shape)
- **SPACE**: Open palm facing down
- **SUBMIT**: Thumbs up

## 📊 Dataset Information

This project integrates with the following datasets for improved recognition:

### 1. Kaggle ASL Datasets
- **ASL Alphabet Dataset**: Individual letter recognition
- **ASL MNIST**: Hand gesture classification
- **ASL Citizen Dataset**: Diverse sign language samples

### 2. How2Sign Dataset
- Large-scale continuous ASL sentence dataset
- Natural signing in context
- Diverse signers and scenarios

### Dataset Attribution
All datasets used in this project are properly attributed. See [DATASETS.md](docs/DATASETS.md) for full licensing information and citations.

## 🏗️ Architecture

### Model Components

#### 1. Hand Tracking (MediaPipe)
- Real-time hand landmark detection
- 21 hand keypoints per hand
- Supports both single and two-hand tracking

#### 2. Gesture Recognition (Basic)
- Rule-based gesture classification from landmarks
- Buffer-based gesture stabilization
- Temporal smoothing

#### 3. Sequence Recognition (Advanced - In Progress)
- LSTM/GRU/Transformer-based architecture
- Temporal attention mechanism
- Sliding window for continuous video
- Sequence-to-sequence translation

#### 4. Natural Language Processing
- ASL grammar rules implementation
- Context-aware interpretation
- Phrase completion and prediction
- Multi-turn conversation tracking

#### 5. Response Generation
- Ollama LLM integration
- Contextual response generation
- Web search integration for real-time information
- ASL-to-English translation

## 📁 Project Structure

```
aslvisionbot/
├── chat.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── config.yaml            # Configuration file
├── .gitignore            # Git ignore rules
├── models/               # Trained model files
│   ├── sequence_model.h5
│   ├── attention_model.h5
│   └── README.md
├── datasets/             # Dataset storage
│   ├── kaggle/
│   ├── how2sign/
│   └── README.md
├── src/                  # Source code modules
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   ├── sequence_model.py
│   ├── asl_grammar.py
│   └── utils.py
├── scripts/              # Utility scripts
│   ├── download_models.py
│   ├── download_datasets.py
│   └── train_model.py
├── tests/               # Unit tests
│   └── test_recognition.py
└── docs/                # Documentation
    ├── DATASETS.md
    ├── MODEL_ARCHITECTURE.md
    └── TRAINING.md
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Dataset paths
- Model parameters
- Camera settings
- LLM configuration
- Processing options

## 🧪 Training Your Own Models

To train models on custom datasets:

```bash
# Prepare your dataset
python scripts/prepare_dataset.py --dataset kaggle

# Train the model
python scripts/train_model.py --config config.yaml --epochs 50

# Evaluate the model
python scripts/evaluate_model.py --model models/sequence_model.h5
```

See [docs/TRAINING.md](docs/TRAINING.md) for detailed training instructions.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

### Dataset Licenses
- Kaggle datasets: Various licenses (see DATASETS.md)
- How2Sign dataset: Research use license
- MediaPipe: Apache 2.0 License

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand tracking capabilities
- **Kaggle** community for ASL datasets
- **How2Sign** team for the continuous ASL dataset
- **Ollama** for local LLM inference
- ASL community for insights and feedback

## 📚 References

1. How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language
2. MediaPipe Hands: On-device Real-time Hand Tracking
3. Various Kaggle ASL datasets and competitions

## 🐛 Troubleshooting

### Camera Not Working
- Check camera permissions
- Try different camera indices (0, 1, 2)
- Ensure no other application is using the camera

### Ollama Connection Error
```bash
# Start Ollama server
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### Model Loading Errors
- Ensure models are downloaded to the correct directory
- Check file permissions
- Verify TensorFlow/Keras installation

### Poor Recognition Accuracy
- Ensure good lighting conditions
- Keep hand within camera frame
- Hold gestures steady for 1+ second
- Check camera is not too close or too far

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Consult the troubleshooting guide

## 🗺️ Roadmap

- [x] Basic hand tracking and gesture recognition
- [x] LLM integration for responses
- [ ] Sentence-level recognition
- [ ] Dataset integration (Kaggle, How2Sign)
- [ ] Advanced sequence modeling
- [ ] ASL grammar support
- [ ] Mobile app version
- [ ] Real-time translation mode
- [ ] Multi-language support

---

**Note**: This project is under active development. Sentence-level recognition and advanced features are being progressively implemented.
