import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve
import os
import logging
from typing import Tuple, Dict
import math
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def step_decay(epoch):
    """Learning rate scheduler"""
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

def plot_training_history(history):
    """
    Plot training and validation loss and accuracy
    
    Args:
        history: History object returned by model.fit()
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name="LSTM"):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random classification
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'models/roc_curve_{model_name.lower()}.png')
    plt.close()

class LSTMModel:
    def __init__(self, input_shape: Tuple[int, int], lstm_units: int = 128, dropout_rate: float = 0.3):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        model = Sequential([
            # First BiLSTM layer with layer normalization
            Bidirectional(LSTM(self.lstm_units, return_sequences=True, 
                             kernel_regularizer=l2(0.01))),
            LayerNormalization(),
            Dropout(self.dropout_rate),
            
            # Second BiLSTM layer
            Bidirectional(LSTM(self.lstm_units, 
                             kernel_regularizer=l2(0.01))),
            LayerNormalization(),
            Dropout(self.dropout_rate),
            
            # Dense layers with regularization
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            LayerNormalization(),
            Dropout(self.dropout_rate),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            LayerNormalization(),
            Dropout(self.dropout_rate),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile with gradient clipping
        optimizer = Adam(clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              batch_size: int = 64, epochs: int = 50) -> tf.keras.callbacks.History:
        # Reshape data for LSTM input
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('models/lstm_best_model.h5', save_best_only=True),
            LearningRateScheduler(step_decay)
        ]
        
        # Train model with class weights
        class_weights = {0: 1, 1: 1}  # Adjust if needed based on class distribution
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        # Reshape test data
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        # Get predictions and probabilities
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Plot ROC curve
        plot_roc_curve(y_test, y_pred_proba, "LSTM")
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        return metrics

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load data
    data_dir = os.path.join('data', 'word2vec_features')
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Initialize and train model
    logger.info("Initializing enhanced LSTM model...")
    model = LSTMModel(input_shape=(1, X_train.shape[1]))
    
    logger.info("Training model...")
    history = model.train(X_train, y_train, X_test, y_test)
    
    # Plot training history
    logger.info("Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    # Print metrics
    logger.info("\nFinal Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 