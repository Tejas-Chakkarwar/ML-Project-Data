"""
Configuration file for LSTM-based EEG-to-Text Classification
Optimized for end-to-end learning without HMM
"""

import os

# ============================================================================
# DATA PARAMETERS (inherited from main config)
# ============================================================================
DATA_DIR = '../processed_data'
SEQUENCE_LENGTH = 5500  # EEG sequence length
MIN_SAMPLES_PER_SENTENCE = 18  # Optimal for 95 classes
TRAIN_TEST_SPLIT = 0.8
NUM_CLASSES = 95  # Will be updated dynamically

# ============================================================================
# LSTM MODEL PARAMETERS
# ============================================================================
INPUT_CHANNELS = 105  # Number of EEG channels
LSTM_HIDDEN_SIZE = 128  # Hidden units in LSTM (can try 64, 128, 256)
LSTM_NUM_LAYERS = 2  # Number of stacked LSTM layers
LSTM_BIDIRECTIONAL = True  # Use bidirectional LSTM
LSTM_DROPOUT = 0.3  # Dropout between LSTM layers

# Attention parameters
USE_ATTENTION = True  # Enable attention mechanism
ATTENTION_HEADS = 4  # Number of attention heads (multi-head attention)

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
BATCH_SIZE = 16  # Batch size for training (adjust based on GPU memory)
EPOCHS = 30  # More epochs for LSTM (they converge slower)
LEARNING_RATE = 1e-3  # Initial learning rate
WEIGHT_DECAY = 1e-4  # L2 regularization

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = 'cosine'  # 'cosine' or 'reduce_on_plateau'
SCHEDULER_PATIENCE = 5  # For ReduceLROnPlateau
SCHEDULER_FACTOR = 0.5  # LR reduction factor

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 10  # Stop if no improvement for N epochs
EARLY_STOP_MIN_DELTA = 0.001  # Minimum improvement to count

# ============================================================================
# DATA AUGMENTATION PARAMETERS
# ============================================================================
NUM_AUGMENTATIONS = 6  # Total samples per original (1 + 5 augmented)
AUG_SCALE_RANGE = (0.8, 1.2)
AUG_NOISE_STD = (0.03, 0.08)
AUG_SHIFT_RANGE = (-50, 50)

# ============================================================================
# MODEL ARCHITECTURE OPTIONS
# ============================================================================
# Input preprocessing
USE_CHANNEL_REDUCTION = True  # Reduce 105 channels using Conv1d
REDUCED_CHANNELS = 32  # Channels after reduction (if enabled)

# Output layer
USE_LABEL_SMOOTHING = True  # Reduce overconfidence
LABEL_SMOOTHING = 0.1  # Smoothing factor

# Gradient clipping
CLIP_GRAD_NORM = 1.0  # Maximum gradient norm

# ============================================================================
# DEVICE AND MEMORY
# ============================================================================
DEVICE = 'cuda'  # 'cuda' or 'cpu'
NUM_WORKERS = 4  # DataLoader workers
PIN_MEMORY = True  # Faster data transfer to GPU

# Memory optimization
GRADIENT_ACCUMULATION_STEPS = 1  # Accumulate gradients over N batches
MIXED_PRECISION = False  # Use FP16 (requires compatible GPU)

# ============================================================================
# CHECKPOINTING AND LOGGING
# ============================================================================
CHECKPOINT_DIR = '../checkpoints'
LSTM_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'lstm_model.pth')
SAVE_BEST_ONLY = True  # Only save model with best validation accuracy

# Logging
LOG_INTERVAL = 10  # Print progress every N batches
SAVE_TRAINING_PLOT = True  # Save loss/accuracy plots

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================
VERBOSE = True
COMPUTE_WORD_LEVEL_METRICS = True  # Compute WER and word accuracy
SAVE_CONFUSION_MATRIX = True

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================
EXPERIMENT_NAME = 'lstm_baseline'  # Used for saving results
SAVE_PREDICTIONS = True  # Save predictions for analysis
