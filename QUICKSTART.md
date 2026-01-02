# Quick Start Guide

Get up and running with ASL Vision Bot in minutes!

## Prerequisites

- Python 3.8 or higher
- Webcam/Camera
- 4GB+ RAM
- (Optional) NVIDIA GPU for model training

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tpC529/aslvisionbot.git
cd aslvisionbot
```

### 2. Choose Your Installation

#### Option A: Basic Mode (Minimal Dependencies)

For basic gesture recognition only:

```bash
# Install minimal dependencies
pip install PyQt6 opencv-python mediapipe requests

# Run the application
python chat.py
```

#### Option B: Enhanced Mode (Full Features)

For sentence-level recognition and all features:

```bash
# Install all dependencies
pip install -r requirements.txt

# Run the application
python chat.py
```

### 3. Start Ollama (Required)

```bash
# Install Ollama from https://ollama.ai

# Start the server
ollama serve

# In another terminal, pull the model
ollama pull llama3.2:1b
```

## First Run

1. **Launch the application**:
   ```bash
   python chat.py
   ```

2. **Camera should start automatically**
   - Green indicators show landmarks on your hand
   - Current gesture displays in the status panel

3. **Try signing**:
   - Sign letter "L" (L-shape with hand)
   - Hold for 1 second
   - You should see "L" appear in "Signing:" field

4. **Complete a word**:
   - Sign: H-E-L-L-O
   - Use thumbs up to submit, or wait 3 seconds
   - Bot will respond!

## Basic Controls

| Action | Gesture |
|--------|---------|
| Letters A-Z | Sign the letter |
| Space | Open palm facing down |
| Submit | Thumbs up |
| Auto-submit | 3 seconds of no gestures |

## Supported Gestures (Basic Mode)

- **A**: Closed fist with thumb alongside
- **B**: Open palm with fingers together
- **L**: Index finger and thumb extended (L-shape)
- **O**: Thumb and index finger forming circle
- **V**: Index and middle fingers extended (V-shape)

More gestures can be added!

## Common Issues

### Camera Not Working

```bash
# Try different camera index
# In chat.py or config.yaml, change:
camera_index = 1  # or 2, 3, etc.
```

### Ollama Not Connecting

```bash
# Make sure Ollama is running
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### Poor Recognition

- Ensure good lighting
- Keep hand in center of frame
- Hold gestures steady for 1 second
- Avoid busy backgrounds

## Next Steps

### Enable Enhanced Mode

```bash
# Install full dependencies
pip install -r requirements.txt

# Download datasets (optional, for training)
python scripts/download_datasets.py

# Application will automatically run in Enhanced mode
python chat.py
```

### Customize Settings

Edit `config.yaml`:

```yaml
camera:
  default_index: 0
  
recognition:
  gesture_buffer_size: 20
  confidence_threshold: 0.7
  
llm:
  model: "llama3.2:1b"
```

### Train Your Own Models

```bash
# Download datasets
python scripts/download_datasets.py

# Train model
python scripts/train_model.py

# Models saved to models/ directory
```

## What's Next?

- Read the [full documentation](README.md)
- Check out [usage examples](docs/USAGE.md)
- Learn about [model architecture](docs/MODEL_ARCHITECTURE.md)
- Explore [training guide](docs/TRAINING.md)
- Review [dataset information](docs/DATASETS.md)

## Getting Help

- Check [troubleshooting](docs/USAGE.md#troubleshooting)
- Open an [issue on GitHub](https://github.com/tpC529/aslvisionbot/issues)
- Review existing issues for solutions

## Example Session

```
1. Start application: python chat.py
2. Wait for camera to initialize (green landmarks appear)
3. Sign: H-E-L-L-O
4. Thumbs up to submit
5. Bot responds: "Hello! How can I help you today?"
6. Continue conversation!
```

## Tips for Best Results

1. **Lighting**: Face a window or use good overhead lighting
2. **Background**: Use a plain, contrasting background
3. **Position**: Keep hand centered, about 2 feet from camera
4. **Speed**: Hold each gesture for a full second
5. **Practice**: Try each gesture a few times to get comfortable

## System Requirements

### Minimum
- Python 3.8+
- 4GB RAM
- Webcam
- CPU: Any modern processor

### Recommended
- Python 3.10+
- 8GB+ RAM
- Good quality webcam (720p or better)
- CPU: Multi-core processor
- (Optional) NVIDIA GPU for training

## Quick Verification

Test your installation:

```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
python -c "import cv2; import mediapipe; import PyQt6; print('OK')"

# Check Ollama
curl http://localhost:11434/api/tags

# Run application
python chat.py
```

If all checks pass, you're ready to go! 🎉

---

Need more help? See the [full README](README.md) or [usage guide](docs/USAGE.md).
