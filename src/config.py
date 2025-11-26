"""
Configuration file for EEG-to-Text HMM Pipeline
Centralized location for all hyperparameters and settings
"""

import os

# ============================================================================
# DATA PARAMETERS
# ============================================================================
DATA_DIR = 'processed_data'
SEQUENCE_LENGTH = 5500  # Target length for padding/truncation
MIN_SAMPLES_PER_SENTENCE = 3  # Minimum samples needed to include a sentence
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
NUM_CLASSES = 344  # Number of unique sentences (will be updated dynamically)

# ============================================================================
# CNN PARAMETERS
# ============================================================================
CNN_INPUT_CHANNELS = 105  # Number of EEG channels
CNN_HIDDEN_CHANNELS = 32  # Feature dimension after encoding
CNN_BATCH_SIZE = 8
CNN_EPOCHS = 5  # Increased from 3 for better feature learning
CNN_LEARNING_RATE = 1e-3
CNN_DEVICE = 'cpu'  # Change to 'cuda' if GPU available
USE_SUPERVISED_CNN = True  # Use supervised CNN instead of autoencoder

# ============================================================================
# DATA AUGMENTATION PARAMETERS
# ============================================================================
NUM_AUGMENTATIONS = 2  # Number of augmented versions per sample
AUG_SCALE_RANGE = (0.8, 1.2)  # Amplitude scaling range (wider range)
AUG_NOISE_STD = (0.03, 0.08)  # Gaussian noise standard deviation range
AUG_SHIFT_RANGE = (-50, 50)  # Time shift range in samples

# ============================================================================
# HMM PARAMETERS
# ============================================================================
HMM_N_STATES = 5  # Increased from 3 for more complex temporal patterns
HMM_N_FEATURES = CNN_HIDDEN_CHANNELS  # Should match CNN output
HMM_MAX_ITER = 10  # Maximum Baum-Welch iterations
HMM_TOLERANCE = 1e-4  # Convergence tolerance
USE_DIAGONAL_COVARIANCE = True  # Use diagonal covariance (more stable)

# ============================================================================
# MODEL CHECKPOINTING
# ============================================================================
CHECKPOINT_DIR = 'checkpoints'
CNN_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'cnn_encoder.pth')
HMM_MODELS_FILE = os.path.join(CHECKPOINT_DIR, 'hmm_models.pkl')

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================
EXTRACT_BATCH_SIZE = 32  # Batch size for feature extraction
VERBOSE = True  # Print detailed progress information
SAVE_CONFUSION_MATRIX = True  # Save confusion matrix plot

# ============================================================================
# QUICK TEST MODE (for fast verification)
# ============================================================================
QUICK_TEST_MAX_FILES = 100  # Max files to load in quick test mode
QUICK_TEST_CNN_EPOCHS = 1  # Reduced epochs for quick test
