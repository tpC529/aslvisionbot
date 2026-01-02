# ASL Vision Bot Models

This directory contains trained models for ASL recognition.

## Model Files

### hand_landmarker.task
- **Type**: MediaPipe Hand Landmarker
- **Purpose**: Hand detection and landmark extraction
- **Source**: Google MediaPipe
- **Size**: ~7.8 MB
- **License**: Apache 2.0

### asl_model.h5
- **Type**: Basic gesture classifier (legacy)
- **Purpose**: Simple static gesture recognition
- **Size**: ~2.7 MB
- **Classes**: A, B, L, O, V, SPACE, SUBMIT, etc.

### sequence_model.h5
- **Type**: LSTM-based sequence model
- **Purpose**: Continuous sign recognition
- **Architecture**: BiLSTM + Attention
- **Input**: (30, 149) - 30 frames, 149 features per frame
- **Output**: Sign probabilities
- **Status**: To be trained

### attention_model.h5
- **Type**: Advanced attention-based model
- **Purpose**: Long sequence recognition
- **Status**: To be trained

## Model Performance

| Model | Task | Accuracy | FPS | Size |
|-------|------|----------|-----|------|
| MediaPipe | Hand Tracking | N/A | 30+ | 7.8 MB |
| Basic Classifier | Static Signs | ~80% | 30 | 2.7 MB |
| Sequence Model | Sign Recognition | TBD | 15-20 | ~20 MB |
| Attention Model | Sentence Recognition | TBD | 10-15 | ~30 MB |

## Download Models

### Pre-trained Models

Basic models are included in the repository. For advanced models:

```bash
# Download from releases
python scripts/download_models.py

# Or manually download from GitHub releases
```

### Train Your Own

```bash
# Download datasets first
python scripts/download_datasets.py

# Train models
python scripts/train_model.py
```

## Model Usage

### Load Model

```python
from tensorflow.keras.models import load_model
from src.sequence_model import AttentionLayer

# Load with custom objects
model = load_model(
    'models/sequence_model.h5',
    custom_objects={'AttentionLayer': AttentionLayer}
)
```

### Inference

```python
import numpy as np

# Prepare input (30 frames of hand features)
features = extract_features(video_frames)
features = np.expand_dims(features, axis=0)  # Add batch dimension

# Predict
predictions = model.predict(features)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)
```

## Model Conversion

### TensorFlow Lite (Mobile)

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('models/sequence_model.h5')

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open('models/sequence_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### ONNX (Cross-platform)

```python
import tf2onnx

# Convert
tf2onnx.convert.from_keras(
    model,
    output_path='models/sequence_model.onnx'
)
```

## Model Updates

Models are versioned and can be updated:

```bash
# Check for updates
python scripts/check_model_updates.py

# Update models
python scripts/update_models.py
```

## Contributing Models

To contribute trained models:

1. Train model following best practices
2. Document architecture and performance
3. Test on validation set
4. Create pull request with model file and documentation
5. Ensure model size is reasonable (<100 MB)

## Storage

Large model files should not be committed directly to Git:

- Use Git LFS for models >10 MB
- Host on GitHub Releases
- Provide download scripts

## License

Models inherit licenses from their training data:
- MediaPipe models: Apache 2.0
- Custom trained models: MIT (if using open datasets)
- Check individual dataset licenses

---

**Note**: This directory structure supports the enhanced ASL Vision Bot with sentence-level recognition. Models are being progressively trained and added.
