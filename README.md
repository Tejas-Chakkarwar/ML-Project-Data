# EEG-to-Text HMM Pipeline

A machine learning pipeline that converts EEG signals to text using a CNN autoencoder for feature extraction and Hidden Markov Models (HMMs) for sentence classification.

## Overview

This project implements a cross-subject EEG-to-text system with the following architecture:

1. **CNN Autoencoder**: Extracts compressed features from raw EEG signals (105 channels × 5500 timepoints)
2. **Data Augmentation**: Generates synthetic training samples using amplitude scaling, noise injection, and time shifting
3. **Gaussian HMMs**: One HMM trained per unique sentence for classification
4. **Cross-Subject Training**: Trains on multiple samples of the same sentence from different subjects

## 🚀 Quick Start on Google Colab

**Want to run this on Google Colab?** Follow these steps:

1. **Upload your dataset to Google Drive**:
   - Create folder: `/MyDrive/ML_Project_Data/processed_data/`
   - Upload all CSV files there

2. **Open the Colab notebook**:
   - Use [`EEG_to_Text_Colab_GitHub.ipynb`](./EEG_to_Text_Colab_GitHub.ipynb)
   - Or follow the [detailed setup guide](./COLAB_GITHUB_SETUP.md)

3. **Run the notebook**:
   - Enable GPU (Runtime → Change runtime type → T4 GPU)
   - Run all cells
   - Training takes 30-60 minutes
   - Expected accuracy: **55-65%**

**The notebook will**:
- Clone this GitHub repository
- Link to your Google Drive data
- Train the model on GPU
- Save results back to your Drive

See [COLAB_GITHUB_SETUP.md](./COLAB_GITHUB_SETUP.md) for detailed instructions.

---

## Project Structure

```
ML Project Data/
├── main.py                 # Main training and evaluation script
├── src/
│   ├── config.py          # Centralized configuration
│   ├── data_loader.py     # Data loading and augmentation
│   ├── feature_extractor.py  # CNN autoencoder
│   ├── hmm_model.py       # Custom Gaussian HMM implementation
│   ├── predictor.py       # Sentence prediction using HMMs
│   └── utils.py           # Evaluation and helper functions
├── processed_data/
│   ├── rawdata_*.csv      # EEG data files (105 × 4760)
│   ├── sentence_mapping.csv   # File-to-sentence mapping
│   └── sentence_mapping.json  # JSON version of mapping
└── checkpoints/           # Saved model checkpoints (created during training)
```

## Requirements

### Python Dependencies

```bash
pip install torch numpy pandas
```

**Required packages:**
- `torch` - PyTorch for CNN autoencoder
- `numpy` - Numerical computations
- `pandas` - Data loading and manipulation

### Data Requirements

- **EEG data files**: CSV files with shape (105, 4760) representing 105 channels and 4760 timepoints
- **Sentence mapping**: CSV/JSON file mapping each data file to its corresponding sentence
- **Minimum samples**: At least 3 samples per sentence for cross-subject training (configurable)

## Quick Start

### Basic Training

Run the full training pipeline with default settings:

```bash
cd "/Users/tejaschakkarwar/Documents/ML Project Data"
python main.py
```

This will:
1. Load all 5,915 data files
2. Filter sentences with ≥3 samples
3. Split into 80% train / 20% test
4. Augment training data (3x multiplication)
5. Train CNN autoencoder (3 epochs)
6. Extract features for HMM
7. Train one HMM per unique sentence
8. Evaluate on test set
9. Save model checkpoints

**Expected time**: 30-60 minutes

### Quick Test Mode

For fast verification (uses only 100 files, 1 epoch):

```bash
python main.py --quick-test
```

**Expected time**: 2-5 minutes

## Command-Line Options

### Mode Options

- `--quick-test` - Run with reduced dataset for fast verification
- `--resume CHECKPOINT` - Resume training from a saved checkpoint

### Data Parameters

- `--min-samples N` - Minimum samples per sentence (default: 3)
- `--train-split RATIO` - Train/test split ratio (default: 0.8)

### CNN Parameters

- `--cnn-epochs N` - Number of CNN training epochs (default: 3)
- `--cnn-batch-size N` - CNN batch size (default: 8)
- `--cnn-lr RATE` - CNN learning rate (default: 0.001)

### HMM Parameters

- `--hmm-states N` - Number of HMM hidden states (default: 3)
- `--hmm-iter N` - Maximum HMM Baum-Welch iterations (default: 10)

### Augmentation

- `--num-aug N` - Number of augmentations per sample (default: 2)

### Output Options

- `--save-models` - Save trained models (default: True)
- `--verbose` - Print detailed progress (default: True)

## Example Commands

### Custom Configuration

```bash
# Train with 5 HMM states and 5 CNN epochs
python main.py --hmm-states 5 --cnn-epochs 5

# Use more augmentation
python main.py --num-aug 5

# Require more samples per sentence
python main.py --min-samples 5
```

### Resume from Checkpoint

```bash
python main.py --resume checkpoints/cnn_encoder.pth
```

## Output

### During Training

The pipeline prints progress for each step:

```
======================================================================
EEG-TO-TEXT HMM PIPELINE
======================================================================

STEP 1: Loading Data
----------------------------------------------------------------------
Loaded mapping file with 5915 entries.
✓ Loaded 5915 sequences

STEP 2: Filtering for Cross-Subject Training
----------------------------------------------------------------------
✓ Found 472 sentences with >= 3 samples

STEP 3: Creating Train/Test Split
----------------------------------------------------------------------
✓ Training Set: 3456 samples
✓ Test Set: 864 samples

...
```

### Final Evaluation

```
======================================================================
EVALUATION SUMMARY
======================================================================
Overall Accuracy: 312/864 (36.11%)

Per-Sentence Accuracy (472 unique sentences):
----------------------------------------------------------------------
  20.0% (1/5) - Henry Ford, with his son Edsel, founded the...
  33.3% (2/6) - After this initial success, Ford left Edison...
  ...

======================================================================
PIPELINE COMPLETED
======================================================================
Total Time: 45m 23s
Final Accuracy: 36.11%
Unique Sentences: 472
Test Samples: 864

Saved Models:
  - CNN Encoder: checkpoints/cnn_encoder.pth
  - HMM Models: checkpoints/hmm_models.pkl
======================================================================
```

## Model Checkpoints

Trained models are automatically saved to the `checkpoints/` directory:

- **`cnn_encoder.pth`** - CNN autoencoder weights
- **`hmm_models.pkl`** - Dictionary of trained HMM models (one per sentence)

## Configuration

Edit `src/config.py` to change default hyperparameters:

```python
# Data parameters
SEQUENCE_LENGTH = 5500
MIN_SAMPLES_PER_SENTENCE = 3

# CNN parameters
CNN_EPOCHS = 3
CNN_BATCH_SIZE = 8
CNN_LEARNING_RATE = 1e-3

# HMM parameters
HMM_N_STATES = 3
HMM_MAX_ITER = 10

# Augmentation
NUM_AUGMENTATIONS = 2
```

## Performance Tips

### Memory Optimization

If you encounter memory issues:
- Reduce `CNN_BATCH_SIZE` in config.py
- Use `--quick-test` mode
- Reduce `--num-aug`

### Improving Accuracy

To potentially improve accuracy:
- Increase `--cnn-epochs` (more CNN training)
- Increase `--hmm-states` (more complex HMMs)
- Increase `--num-aug` (more data augmentation)
- Increase `--min-samples` (better HMM training, but fewer sentences)

### Speed Optimization

To train faster:
- Reduce `--cnn-epochs`
- Reduce `--num-aug`
- Use `--quick-test` for verification

## Troubleshooting

### "No sentences found with enough samples"

**Solution**: Lower `--min-samples` or ensure you have multiple samples per sentence in your dataset.

### Memory errors during training

**Solution**: Reduce batch size with `--cnn-batch-size 4` or use `--quick-test` mode.

### Low accuracy

**Expected**: Accuracy depends heavily on:
- Number of unique sentences (more sentences = harder task)
- Quality of EEG signals
- Number of samples per sentence
- Typical range: 20-50% for hundreds of sentences

## Citation

If you use this code, please cite the relevant EEG-to-text research papers and acknowledge the custom HMM implementation.

## License

[Add your license information here]
