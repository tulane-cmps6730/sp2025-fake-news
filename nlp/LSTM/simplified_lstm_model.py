import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
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
    """Learning rate scheduler with gentler decay"""
    initial_lr = 0.0005  # Reduced initial learning rate
    drop = 0.7  # Gentler decay
    epochs_drop = 15.0  # Longer intervals between drops
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

def plot_training_history(history):
    """Plot training and validation metrics"""
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
    
    plt.tight_layout()
    plt.savefig('models/simplified_training_history.png')
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name="Simplified LSTM"):
    """Plot ROC curve for the model"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'models/roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

class SimplifiedLSTMModel:
    def __init__(self, input_shape: Tuple[int, int], lstm_units: int = 128, dropout_rate: float = 0.4):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        model = Sequential([
            # First BiLSTM layer with stronger regularization
            Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                             kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                             recurrent_regularizer=l1_l2(l1=0.01, l2=0.01))),
            LayerNormalization(),
            Dropout(self.dropout_rate),
            
            # Second BiLSTM layer
            Bidirectional(LSTM(self.lstm_units,
                             kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                             recurrent_regularizer=l1_l2(l1=0.01, l2=0.01))),
            LayerNormalization(),
            Dropout(self.dropout_rate),
            
            # Dense layer with regularization
            Dense(64, activation='relu',
                  kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            LayerNormalization(),
            Dropout(self.dropout_rate),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile with gradient clipping and reduced learning rate
        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
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
        
        # Enhanced callbacks with more patience
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint('models/simplified_lstm_best_model.h5', save_best_only=True),
            LearningRateScheduler(step_decay)
        ]
        
        # Calculate class weights to handle imbalance
        class_weights = self._calculate_class_weights(y_train)
        
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
    
    def _calculate_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """Calculate class weights to handle imbalance"""
        total = len(y_train)
        pos = np.sum(y_train)
        neg = total - pos
        
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        
        return {0: weight_for_0, 1: weight_for_1}
    
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
        plot_roc_curve(y_test, y_pred_proba, "Simplified LSTM")
        
        # Print classification report
        report = classification_report(y_test, y_pred)
        logger.info("\nClassification Report:\n%s", report)
        
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
    logger.info("Initializing simplified LSTM model...")
    model = SimplifiedLSTMModel(input_shape=(1, X_train.shape[1]))
    
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