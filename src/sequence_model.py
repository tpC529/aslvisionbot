"""
Sequence model for ASL sentence recognition.

This module implements LSTM/GRU-based models with attention mechanisms
for recognizing continuous ASL sentences.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """Attention mechanism for sequence models"""
    
    def __init__(self, units: int, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        uit = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        attention_weights = tf.nn.softmax(ait, axis=1)
        attention_weights = tf.expand_dims(attention_weights, -1)
        weighted_input = x * attention_weights
        return tf.reduce_sum(weighted_input, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class SequenceEncoder(keras.Model):
    """Encoder for video sequences using LSTM/GRU"""
    
    def __init__(self, lstm_units: int = 128, dropout_rate: float = 0.3, **kwargs):
        super(SequenceEncoder, self).__init__(**kwargs)
        
        self.lstm1 = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True)
        )
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.lstm2 = layers.Bidirectional(
            layers.LSTM(lstm_units // 2, return_sequences=True)
        )
        self.dropout2 = layers.Dropout(dropout_rate)
        
        self.attention = AttentionLayer(lstm_units)
        
    def call(self, x, training=False):
        # x shape: (batch_size, time_steps, features)
        x = self.lstm1(x)
        x = self.dropout1(x, training=training)
        
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        
        # Apply attention
        x = self.attention(x)
        
        return x


class ASLSequenceModel:
    """Complete model for ASL sentence recognition"""
    
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (sequence_length, feature_dim)
        num_classes: int,
        lstm_units: int = 128,
        dropout_rate: float = 0.3,
        use_attention: bool = True
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build the sequence recognition model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # LSTM layers
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True)
        )(inputs)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units // 2, return_sequences=True)
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        if self.use_attention:
            # Apply attention mechanism
            x = AttentionLayer(self.lstm_units)(x)
        else:
            # Use last output
            x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='asl_sequence_model')
        
        return model
    
    def compile_model(
        self,
        learning_rate: float = 0.001,
        loss: str = 'categorical_crossentropy'
    ):
        """Compile the model with optimizer and loss"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        logger.info(f"Model compiled with lr={learning_rate}, loss={loss}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: Optional[list] = None
    ):
        """Train the model"""
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X, batch_size=batch_size)
    
    def save(self, filepath: str):
        """Save model to file"""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        logger.info(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()


class TemporalSequenceProcessor:
    """Processes temporal sequences for continuous sign recognition"""
    
    def __init__(
        self,
        model: ASLSequenceModel,
        sequence_length: int = 30,
        stride: int = 10,
        confidence_threshold: float = 0.7
    ):
        self.model = model
        self.sequence_length = sequence_length
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.frame_buffer = []
    
    def add_frame(self, features: np.ndarray):
        """Add a frame's features to the buffer"""
        self.frame_buffer.append(features)
        
        # Keep only the most recent frames
        if len(self.frame_buffer) > self.sequence_length * 2:
            self.frame_buffer = self.frame_buffer[-self.sequence_length * 2:]
    
    def process_continuous_sequence(self) -> Optional[Tuple[str, float]]:
        """
        Process the current frame buffer and return detected sign if any.
        
        Returns:
            (sign_label, confidence) or None
        """
        if len(self.frame_buffer) < self.sequence_length:
            return None
        
        # Get the last sequence_length frames
        sequence = np.array(self.frame_buffer[-self.sequence_length:])
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
        
        # Predict
        predictions = self.model.predict(sequence)[0]
        confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        if confidence >= self.confidence_threshold:
            return (str(predicted_class), confidence)
        
        return None
    
    def reset(self):
        """Clear the frame buffer"""
        self.frame_buffer = []


def create_callbacks(checkpoint_dir: str = 'checkpoints/') -> list:
    """Create training callbacks"""
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f'{checkpoint_dir}/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='logs/tensorboard',
            histogram_freq=1
        )
    ]
    
    return callbacks


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = ASLSequenceModel(
        input_shape=(30, 63),  # 30 frames, 63 features (21 landmarks * 3 coords)
        num_classes=50,  # Example: 50 different signs
        lstm_units=128,
        dropout_rate=0.3,
        use_attention=True
    )
    
    model.compile_model()
    model.summary()
    
    print("\nModel created successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output classes: {model.num_classes}")
