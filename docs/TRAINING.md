# Model Training Guide

This guide provides detailed instructions for training ASL recognition models.

## Prerequisites

1. **Datasets Downloaded**: Ensure you have downloaded the required datasets
   ```bash
   python scripts/download_datasets.py
   ```

2. **Dependencies Installed**: All Python packages installed
   ```bash
   pip install -r requirements.txt
   ```

3. **Hardware**: 
   - Minimum: 8GB RAM, CPU
   - Recommended: 16GB+ RAM, NVIDIA GPU with CUDA

## Quick Start

### 1. Basic Training

Train a model with default settings:

```bash
python scripts/train_model.py
```

### 2. Custom Configuration

Edit `config.yaml` to customize training:

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  validation_split: 0.2
```

### 3. Monitor Training

View training progress in TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

## Training Pipeline

### Step 1: Data Preparation

```python
from src.dataset_loader import DatasetLoader
from src.preprocessing import VideoPreprocessor

# Load datasets
loader = DatasetLoader(config['datasets'])
images, labels, classes = loader.load_alphabet_dataset('kaggle')

# Preprocess
preprocessor = VideoPreprocessor(
    target_size=(224, 224),
    normalize=True
)
processed = preprocessor.preprocess_images(images)
```

### Step 2: Create Model

```python
from src.sequence_model import ASLSequenceModel

model = ASLSequenceModel(
    input_shape=(30, 149),  # 30 frames, 149 features
    num_classes=50,
    lstm_units=128,
    dropout_rate=0.3,
    use_attention=True
)

model.compile_model(learning_rate=0.001)
```

### Step 3: Train Model

```python
from src.sequence_model import create_callbacks

callbacks = create_callbacks(checkpoint_dir='checkpoints/')

history = model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)
```

### Step 4: Evaluate Model

```python
# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2%}")

# Make predictions
predictions = model.predict(X_test)
```

### Step 5: Save Model

```python
model.save('models/sequence_model.h5')
```

## Advanced Training

### Transfer Learning

Use pre-trained models as starting point:

```python
from tensorflow.keras.models import load_model

# Load pre-trained model
base_model = load_model('models/pretrained_base.h5')

# Freeze base layers
for layer in base_model.layers[:-5]:
    layer.trainable = False

# Fine-tune on your data
model.train(X_train, y_train, epochs=20)
```

### Data Augmentation

Apply augmentation during training:

```python
from src.preprocessing import DataAugmentation

augmenter = DataAugmentation(
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=(0.8, 1.2)
)

# Augment during training
for epoch in range(epochs):
    for batch in data_loader:
        augmented = augmenter.augment_batch(batch)
        model.train_on_batch(augmented)
```

### Multi-GPU Training

Distribute training across multiple GPUs:

```python
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = ASLSequenceModel(...)
    model.compile_model()

model.train(X_train, y_train, epochs=50)
```

## Training for Different Tasks

### 1. Alphabet Recognition (Static Signs)

```python
# Use image dataset
X, y = loader.load_alphabet_dataset('kaggle')

# Simple CNN model
model = create_cnn_model(num_classes=26)
model.train(X, y, epochs=30)
```

### 2. Word Recognition (Short Sequences)

```python
# Use short video clips
sequences, labels = loader.load_word_dataset()

# LSTM model
model = ASLSequenceModel(
    input_shape=(15, 149),  # 15 frames
    num_classes=500  # 500 words
)
model.train(sequences, labels, epochs=50)
```

### 3. Sentence Recognition (Long Sequences)

```python
# Use How2Sign dataset
video_sequences, sentences = loader.load_sentence_dataset('how2sign')

# Sequence-to-sequence model
model = ASLSeq2SeqModel(
    input_shape=(60, 149),  # 60 frames
    vocab_size=5000
)
model.train(video_sequences, sentences, epochs=100)
```

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'lstm_units': [64, 128, 256],
    'dropout_rate': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.0001]
}

# Note: Requires scikit-learn wrapper for Keras
best_model = grid_search(param_grid, X_train, y_train)
```

### Random Search

```python
import optuna

def objective(trial):
    lstm_units = trial.suggest_int('lstm_units', 64, 256)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    
    model = ASLSequenceModel(
        input_shape=(30, 149),
        num_classes=50,
        lstm_units=lstm_units,
        dropout_rate=dropout
    )
    model.compile_model(learning_rate=lr)
    
    history = model.train(X_train, y_train, epochs=20)
    return history.history['val_accuracy'][-1]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

## Training Tips

### 1. Start Small
- Begin with a subset of data
- Use fewer epochs
- Verify pipeline works end-to-end

### 2. Monitor Overfitting
- Watch validation loss
- Use early stopping
- Increase dropout if needed

### 3. Learning Rate Schedule
- Start with higher learning rate
- Reduce on plateau
- Use warmup for first few epochs

### 4. Batch Size
- Larger batch: faster, less stable
- Smaller batch: slower, more stable
- Try 16, 32, 64

### 5. Regularization
- Dropout: 0.2-0.4
- L2 regularization: 1e-4
- Data augmentation

## Troubleshooting

### Low Accuracy
- Check data quality
- Verify labels are correct
- Increase model capacity
- Train for more epochs
- Add data augmentation

### Overfitting
- Reduce model size
- Increase dropout
- Add L2 regularization
- Use more training data
- Reduce training time

### Slow Training
- Reduce batch size
- Use GPU acceleration
- Optimize data loading
- Use mixed precision training
- Cache preprocessed data

### Out of Memory
- Reduce batch size
- Reduce model size
- Use gradient accumulation
- Clear unused variables

## Benchmarking

### Dataset Splits

```python
# Standard split
train: 70%
validation: 15%
test: 15%

# Cross-validation
5-fold or 10-fold CV
```

### Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix

# Classification report
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Per-class accuracy
per_class = cm.diagonal() / cm.sum(axis=1)
```

## Deployment

### Export Model

```python
# Save as HDF5
model.save('models/production_model.h5')

# Save as SavedModel (TF Serving)
model.model.save('models/saved_model/')

# Convert to TFLite (mobile)
converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
tflite_model = converter.convert()

# Save ONNX (cross-platform)
import tf2onnx
tf2onnx.convert.from_keras(model.model, output_path='models/model.onnx')
```

### Optimize for Inference

```python
# Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Pruning
import tensorflow_model_optimization as tfmot

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model.model)
```

## Resources

- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Guide: https://keras.io/guides/
- ASL Datasets: See docs/DATASETS.md
- Model Architecture: See docs/MODEL_ARCHITECTURE.md

---

For questions or issues, please open a GitHub issue or consult the documentation.
