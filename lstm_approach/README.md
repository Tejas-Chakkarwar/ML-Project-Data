# LSTM-Based EEG-to-Text Classification

This folder contains an alternative approach to EEG-to-text classification using **end-to-end LSTM** with attention mechanism, replacing the CNN+HMM pipeline.

## Overview

### Why LSTM?

The original CNN+HMM approach has limitations:
- **HMM bottleneck**: Hidden Markov Models struggle with complex temporal patterns
- **Two-stage pipeline**: CNN feature extraction and HMM classification are trained separately
- **Limited temporal modeling**: HMMs assume Markov property (limited memory)

LSTM advantages:
- **End-to-end learning**: Single model trained with backpropagation
- **Long-range dependencies**: LSTMs can capture patterns across long sequences
- **Attention mechanism**: Focus on important time steps automatically
- **Better for sequential data**: Designed for time-series like EEG

### Architecture Comparison

```
CNN+HMM Approach:
EEG Input (105×5500)
    ↓
CNN Encoder (supervised)
    ↓
Features (32×688)
    ↓
HMM (per sentence)
    ↓
Prediction

LSTM Approach:
EEG Input (105×5500)
    ↓
Channel Reduction (optional): 105 → 32 channels
    ↓
Bidirectional LSTM (128 hidden units × 2 layers)
    ↓
Multi-head Attention (4 heads)
    ↓
Global Pooling (avg + max)
    ↓
Fully Connected Classifier
    ↓
Prediction
```

## File Structure

```
lstm_approach/
├── config_lstm.py       # LSTM-specific configuration
├── lstm_model.py        # Model architecture with attention
├── train_lstm.py        # Training script
└── README.md           # This file
```

## Model Architecture

### Key Components

1. **Channel Reduction** (Optional)
   - Reduces 105 EEG channels to 32 using 1D convolutions
   - Reduces computational cost and parameters
   - Can be disabled with `--no-channel-reduction`

2. **Bidirectional LSTM**
   - 2 stacked layers with 128 hidden units each
   - Bidirectional: processes sequence forward and backward
   - Dropout for regularization

3. **Multi-head Attention**
   - 4 attention heads to focus on important time steps
   - Helps model learn which EEG periods are critical
   - Can be disabled with `--no-attention`

4. **Global Pooling**
   - Combines average and max pooling
   - Aggregates temporal information into fixed-size vector

5. **Classification Head**
   - 3-layer fully connected network
   - Dropout for regularization
   - Outputs class probabilities

### Model Size

With default configuration (95 classes):
- Total parameters: ~1.5-2 million (depending on options)
- Much larger than CNN alone, but trains end-to-end
- GPU recommended but can run on CPU

## Usage

### Basic Training

```bash
cd lstm_approach
python train_lstm.py
```

This will:
1. Load data from `../processed_data/`
2. Filter sentences with ≥18 samples (95 classes)
3. Split into train (70%), validation (15%), test (15%)
4. Train for 30 epochs with early stopping
5. Save best model to `../checkpoints/lstm_model.pth`
6. Evaluate on test set with word-level metrics

### Configuration Options

```bash
# Quick test with reduced data
python train_lstm.py --quick-test

# Train with 5 classes (for better demo results)
python train_lstm.py --min-samples 20

# Train on CPU
python train_lstm.py --device cpu

# Disable attention mechanism
python train_lstm.py --no-attention

# Disable channel reduction (use all 105 channels)
python train_lstm.py --no-channel-reduction

# Custom hyperparameters
python train_lstm.py --hidden-size 256 --num-layers 3 --batch-size 32 --epochs 50

# Disable early stopping
python train_lstm.py --no-early-stop

# More augmentation
python train_lstm.py --num-aug 8
```

### Advanced Options

```bash
# Custom data split
python train_lstm.py --train-split 0.8 --val-split 0.1

# Custom learning rate and weight decay
python train_lstm.py --lr 0.0005 --weight-decay 0.0001

# Resume from checkpoint
python train_lstm.py --load-checkpoint ../checkpoints/lstm_model.pth
```

## Configuration File

Edit `config_lstm.py` to change:

### Model Architecture
- `LSTM_HIDDEN_SIZE`: LSTM hidden units (default: 128)
- `LSTM_NUM_LAYERS`: Number of LSTM layers (default: 2)
- `LSTM_BIDIRECTIONAL`: Use bidirectional LSTM (default: True)
- `USE_ATTENTION`: Enable attention mechanism (default: True)
- `ATTENTION_HEADS`: Number of attention heads (default: 4)
- `USE_CHANNEL_REDUCTION`: Reduce input channels (default: True)
- `REDUCED_CHANNELS`: Channels after reduction (default: 32)

### Training Parameters
- `BATCH_SIZE`: Batch size (default: 16)
- `EPOCHS`: Maximum epochs (default: 30)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)
- `WEIGHT_DECAY`: L2 regularization (default: 0.0001)

### Regularization
- `LSTM_DROPOUT`: Dropout in LSTM (default: 0.3)
- `USE_LABEL_SMOOTHING`: Prevent overconfidence (default: True)
- `LABEL_SMOOTHING`: Smoothing factor (default: 0.1)
- `CLIP_GRAD_NORM`: Gradient clipping (default: 1.0)

### Early Stopping
- `USE_EARLY_STOPPING`: Enable early stopping (default: True)
- `EARLY_STOP_PATIENCE`: Patience epochs (default: 10)
- `EARLY_STOP_MIN_DELTA`: Minimum improvement (default: 0.001)

## Expected Results

Based on the dataset characteristics:

### 5-Class Model (min_samples=20)
- **Expected accuracy**: 60-80%
- **Training time**: ~10-20 minutes (GPU) / ~1-2 hours (CPU)
- **Random baseline**: 20%
- **Use case**: Demo and proof of concept

### 95-Class Model (min_samples=18)
- **Expected accuracy**: 8-15%
- **Training time**: ~30-60 minutes (GPU) / ~3-6 hours (CPU)
- **Random baseline**: 1.05%
- **Improvement over CNN+HMM**: 2-3x better
- **Use case**: Research and real-world scenario

### Why LSTM Should Outperform CNN+HMM

1. **End-to-end optimization**: Entire model trained jointly
2. **Better temporal modeling**: LSTMs excel at sequences
3. **Attention mechanism**: Focuses on relevant time periods
4. **No HMM limitations**: No Markov assumption
5. **Gradient flow**: Backpropagation through entire model

## Output Files

Training produces:

1. **Model checkpoint**: `../checkpoints/lstm_model.pth`
   - Best model based on validation accuracy
   - Includes model weights, optimizer state, and label mapping

2. **Training plot**: `../checkpoints/lstm_training_history.png`
   - Loss and accuracy curves for train and validation

3. **Console output**: Detailed metrics including word-level accuracy and WER

## Testing the Model

To test the model architecture without training:

```bash
python lstm_model.py
```

This will:
- Create a model with default parameters
- Print model summary (architecture and parameter count)
- Run a test forward pass with dummy data
- Verify output shapes are correct

## Comparison with CNN+HMM

| Aspect | CNN+HMM | LSTM |
|--------|---------|------|
| **Training** | Two-stage | End-to-end |
| **Temporal modeling** | HMM (Markov) | LSTM (long memory) |
| **Parameters** | ~500K (CNN) + HMMs | ~1.5-2M total |
| **Training time** | Faster | Slower |
| **Accuracy (95 classes)** | 4-7% | 8-15% (expected) |
| **Accuracy (5 classes)** | 60-80% | 70-85% (expected) |
| **Interpretability** | HMM states | Attention weights |
| **GPU requirement** | Optional | Recommended |

## Troubleshooting

### Out of Memory (GPU)

```bash
# Reduce batch size
python train_lstm.py --batch-size 8

# Disable channel reduction (counterintuitively can help)
python train_lstm.py --no-channel-reduction --batch-size 8

# Use CPU (slower)
python train_lstm.py --device cpu
```

### Out of Memory (RAM)

```bash
# Reduce augmentation
python train_lstm.py --num-aug 2

# Quick test mode
python train_lstm.py --quick-test
```

### Poor Accuracy

```bash
# More epochs and larger model
python train_lstm.py --epochs 50 --hidden-size 256 --num-layers 3

# More augmentation
python train_lstm.py --num-aug 8

# Disable label smoothing
# Edit config_lstm.py: USE_LABEL_SMOOTHING = False

# Try without attention (sometimes simpler is better)
python train_lstm.py --no-attention
```

### Training Too Slow

```bash
# Use GPU
python train_lstm.py --device cuda

# Increase batch size (if GPU has memory)
python train_lstm.py --batch-size 32

# Reduce model size
python train_lstm.py --hidden-size 64 --num-layers 1

# Fewer augmentations
python train_lstm.py --num-aug 2
```

## Next Steps

After training and evaluating:

1. **Compare with CNN+HMM**: Check if LSTM achieves higher accuracy
2. **Analyze attention weights**: See which time steps are important
3. **Error analysis**: Use `analyze_predictions.py` (to be created)
4. **Hyperparameter tuning**: Try different architectures
5. **Ensemble**: Combine LSTM and CNN+HMM predictions

## Future Improvements

Potential enhancements:
- **Transformer architecture**: Replace LSTM with self-attention
- **Pre-training**: Learn general EEG features on larger dataset
- **Word-level classification**: Predict words instead of sentences
- **Multi-task learning**: Combine with other EEG tasks
- **Data augmentation**: More sophisticated techniques (mixup, etc.)
- **Model compression**: Knowledge distillation for deployment

## References

- Bidirectional LSTM: Schuster & Paliwal (1997)
- Attention mechanism: Bahdanau et al. (2014)
- Multi-head attention: Vaswani et al. (2017)

---

For questions or issues, refer to the main project README or check the training logs.
