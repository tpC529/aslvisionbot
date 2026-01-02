# ASL Vision Bot - Usage Examples

This guide provides practical examples of using the ASL Vision Bot with sentence-level recognition.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Enhanced Mode](#enhanced-mode)
3. [Dataset Integration](#dataset-integration)
4. [Model Training](#model-training)
5. [Custom Configuration](#custom-configuration)
6. [API Usage](#api-usage)
7. [Troubleshooting](#troubleshooting)

## Basic Usage

### Running the Application

```bash
# Basic mode (no additional dependencies)
python chat.py

# With enhanced features (requires dependencies)
pip install -r requirements.txt
python chat.py
```

### Signing with the Application

1. **Start the application**: Camera starts automatically
2. **Position your hand**: Keep hand clearly visible in frame
3. **Sign letters**: Hold each letter gesture for ~1 second
4. **Add spaces**: Use open palm facing down
5. **Submit**: Use thumbs up or wait 3 seconds
6. **View response**: Bot responds to your message

### Example Conversation Flow

```
User signs: H-E-L-L-O
Bot: "Hello! How can I help you today?"

User signs: W-H-A-T SPACE I-S SPACE A-S-L
Bot: "ASL stands for American Sign Language. It's a complete, 
      natural language used by the Deaf community."

User signs: T-H-A-N-K SPACE Y-O-U
Bot: "You're welcome! Feel free to ask me anything else."
```

## Enhanced Mode

### What is Enhanced Mode?

Enhanced mode activates when all dependencies are installed. It provides:
- **Sentence-level recognition**: Better understanding of full sentences
- **ASL grammar interpretation**: Proper translation to English grammar
- **Context awareness**: Remembers previous conversation
- **Phrase prediction**: Suggests likely next signs
- **Better accuracy**: Uses trained models for recognition

### Activating Enhanced Mode

```bash
# Install all dependencies
pip install -r requirements.txt

# Run the application
python chat.py

# You'll see "Enhanced Mode" in the window title
```

### Enhanced Features in Action

#### ASL Grammar Translation

```python
# Input signs (ASL order): TOMORROW I GO SCHOOL
# Basic mode output: "TOMORROWIGOSPACE"
# Enhanced mode output: "I will go to school tomorrow."
```

#### Context-Aware Responses

```python
# Conversation 1
User: "WHAT WEATHER TODAY"
Bot: "It's sunny and 75 degrees."

# Conversation 2 (context-aware)
User: "TOMORROW"  # Short follow-up
Bot: "Tomorrow is expected to be partly cloudy with a high of 72."
```

#### Phrase Completion

```python
# User signs: "HELLO MY NAME"
# Bot predicts: "IS" as likely next sign
# Helps with faster input
```

## Dataset Integration

### Downloading Datasets

#### Kaggle Datasets

```bash
# Setup Kaggle API
pip install kaggle
# Place kaggle.json in ~/.kaggle/

# Download datasets
python scripts/download_datasets.py
```

#### How2Sign Dataset

1. Visit: https://how2sign.github.io/
2. Fill out data request form
3. Download after approval
4. Extract to `datasets/how2sign/`

### Using Datasets

```python
from src.dataset_loader import DatasetLoader

# Initialize loader
config = {
    'kaggle': {'path': 'datasets/kaggle/'},
    'how2sign': {'path': 'datasets/how2sign/'},
    'preprocessing': {
        'target_size': [224, 224],
        'normalize': True
    }
}

loader = DatasetLoader(config)

# Load alphabet dataset
images, labels, classes = loader.load_alphabet_dataset('kaggle')
print(f"Loaded {len(images)} images with {len(classes)} classes")

# Load sentence dataset
videos, sentences = loader.load_sentence_dataset('how2sign', split='train')
print(f"Loaded {len(videos)} video sequences")
```

## Model Training

### Training a Sequence Model

```bash
# Download datasets first
python scripts/download_datasets.py

# Train the model
python scripts/train_model.py

# Model will be saved to models/sequence_model.h5
```

### Custom Training Script

```python
from src.sequence_model import ASLSequenceModel, create_callbacks
from src.dataset_loader import DatasetLoader

# Load data
loader = DatasetLoader(config)
X_train, y_train = loader.load_training_data()

# Create model
model = ASLSequenceModel(
    input_shape=(30, 149),  # 30 frames, 149 features
    num_classes=50,
    lstm_units=128,
    use_attention=True
)

# Compile and train
model.compile_model(learning_rate=0.001)
callbacks = create_callbacks()

history = model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# Save model
model.save('models/my_custom_model.h5')
```

### Evaluating Model Performance

```python
# Load trained model
from tensorflow.keras.models import load_model
from src.sequence_model import AttentionLayer

model = load_model(
    'models/sequence_model.h5',
    custom_objects={'AttentionLayer': AttentionLayer}
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2%}")

# Make predictions
predictions = model.predict(X_test)
```

## Custom Configuration

### Editing config.yaml

```yaml
# Camera settings
camera:
  default_index: 0
  fps: 30

# Model settings
models:
  sequence_model: "models/custom_sequence_model.h5"
  lstm_units: 256  # Increase for larger model
  
# Recognition settings
recognition:
  gesture_buffer_size: 30  # More buffering
  confidence_threshold: 0.8  # Higher threshold
  
# LLM settings
llm:
  model: "llama3.2:3b"  # Use larger model
  temperature: 0.5  # Less creative responses
```

### Loading Custom Configuration

```python
from src.utils import load_config

# Load configuration
config = load_config('my_custom_config.yaml')

# Use in application
from chat import ASLChatbotApp

app = ASLChatbotApp()
# Config is loaded automatically if config.yaml exists
```

## API Usage

### Using ASL Interpreter Programmatically

```python
from src.asl_grammar import ASLInterpreter

# Initialize interpreter
interpreter = ASLInterpreter()

# Interpret sign sequence
signs = ['HELLO', 'MY', 'NAME', 'IS', 'JOHN']
result = interpreter.interpret_signs(signs)

print(f"Signs: {result['signs']}")
print(f"English: {result['english']}")
print(f"Is Question: {result['is_question']}")
print(f"Expression Type: {result['expression_type']}")

# Output:
# Signs: ['HELLO', 'MY', 'NAME', 'IS', 'JOHN']
# English: Hello, my name is John.
# Is Question: False
# Expression Type: greeting
```

### Using Sequence Model for Prediction

```python
from src.sequence_model import TemporalSequenceProcessor
import numpy as np

# Initialize processor
processor = TemporalSequenceProcessor(
    model=trained_model,
    sequence_length=30,
    confidence_threshold=0.7
)

# Process frames
for frame_features in video_frames:
    processor.add_frame(frame_features)
    
    # Check for detection
    result = processor.process_continuous_sequence()
    if result:
        sign_label, confidence = result
        print(f"Detected: {sign_label} (confidence: {confidence:.2f})")
```

### Context Management

```python
from src.asl_grammar import ContextManager

# Initialize context manager
context = ContextManager(context_window=5)

# Add conversation exchanges
context.add_exchange(
    "Hello, how are you?",
    "I'm doing well, thank you!"
)

# Get formatted context
context_text = context.get_context()
print(context_text)

# Check for follow-up questions
is_followup = context.is_follow_up_question(['WHY'])
print(f"Is follow-up: {is_followup}")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Camera Not Working

**Problem**: "Cannot access camera"

**Solutions**:
```bash
# Check camera permissions
# Linux: Add user to video group
sudo usermod -a -G video $USER

# Try different camera indices
# Edit config.yaml:
camera:
  default_index: 1  # or 2, 3, etc.
```

#### 2. Ollama Connection Error

**Problem**: "Cannot connect to Ollama"

**Solutions**:
```bash
# Start Ollama server
ollama serve

# In another terminal, pull model
ollama pull llama3.2:1b

# Verify it's running
curl http://localhost:11434/api/tags
```

#### 3. Poor Recognition Accuracy

**Problem**: Signs not recognized correctly

**Solutions**:
- Ensure good lighting
- Keep hand centered in frame
- Hold gestures steady for 1+ seconds
- Avoid cluttered background
- Check hand_landmarker.task is present

#### 4. Module Import Errors

**Problem**: "No module named 'src'"

**Solutions**:
```bash
# Make sure you're in the project directory
cd /path/to/aslvisionbot

# Run from project root
python chat.py

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/aslvisionbot"
```

#### 5. Out of Memory During Training

**Problem**: "OOM when allocating tensor"

**Solutions**:
```yaml
# Edit config.yaml
training:
  batch_size: 16  # Reduce from 32
  
models:
  lstm_units: 64  # Reduce from 128
```

#### 6. Enhanced Mode Not Activating

**Problem**: Running in Basic mode despite dependencies

**Solutions**:
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify imports work
python -c "from src.asl_grammar import ASLInterpreter; print('OK')"

# Check for import errors in console
python chat.py
# Look for "Running in ENHANCED mode" message
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or in code:
```python
# Edit chat.py
logging.basicConfig(level=logging.DEBUG)  # Change from INFO
```

### Performance Monitoring

```python
from src.utils import PerformanceMonitor

monitor = PerformanceMonitor()

# Track metrics
monitor.add_inference_time(15.3)
monitor.add_frame_rate(28.5)

# Print statistics
monitor.print_stats()
```

## Advanced Usage

### Custom Sign Recognition

Add your own gesture recognition rules:

```python
# In chat.py, ASLRecognizer class
def _analyze_hand_pose(self, landmarks):
    # ... existing code ...
    
    # Add custom gesture
    custom_gesture = (
        # Your custom logic here
        condition1 and condition2
    )
    
    if custom_gesture:
        return 'CUSTOM_SIGN'
```

### Integration with Other Systems

```python
# Use as a library
from chat import ASLRecognizer

recognizer = ASLRecognizer()

# Process your own video frames
for frame in your_video:
    processed_frame, gesture, text = recognizer.process_frame(frame)
    
    if recognizer.is_sentence_complete():
        sentence = recognizer.get_and_clear_text()
        # Send to your system
        your_system.process(sentence)
```

## Next Steps

1. **Explore the Code**: Check out `src/` modules
2. **Train Models**: Download datasets and train
3. **Customize**: Edit config.yaml for your needs
4. **Contribute**: Add new features and submit PRs

## Resources

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory (coming soon)
- **Issues**: GitHub Issues page
- **Community**: Join discussions on GitHub

---

For more information, see the full documentation in the `docs/` directory.
