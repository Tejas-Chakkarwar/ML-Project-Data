# Training and Testing Guide

Complete guide for training and evaluating the EEG-to-Text HMM pipeline.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Training Workflow](#training-workflow)
3. [Testing and Evaluation](#testing-and-evaluation)
4. [Understanding Results](#understanding-results)
5. [Advanced Usage](#advanced-usage)
6. [Tips for Improving Performance](#tips-for-improving-performance)

---

## Prerequisites

### Data Preparation Checklist

Before running training, ensure:

- [ ] All CSV data files are in `processed_data/` directory
- [ ] Files have shape (105, 4760) - 105 channels, 4760 timepoints
- [ ] `sentence_mapping.csv` exists and maps each file to its sentence
- [ ] No NaN-only files remain (run cleanup if needed)
- [ ] At least 3 samples per sentence for cross-subject training

### Environment Setup

```bash
# Install dependencies
pip install torch numpy pandas

# Verify installation
python -c "import torch; import pandas; import numpy; print('Dependencies OK')"
```

---

## Training Workflow

### Step 1: Quick Verification (Recommended First Step)

Before running full training, verify everything works:

```bash
python main.py --quick-test
```

**What it does:**
- Uses only 100 files (fast)
- Trains CNN for 1 epoch
- Completes in 2-5 minutes

**Expected output:**
```
======================================================================
EEG-TO-TEXT HMM PIPELINE
======================================================================
⚡ Quick test mode: using 100 files
...
✓ Loaded 100 sequences
...
Final Accuracy: ~25-35%
```

If this completes successfully, proceed to full training.

### Step 2: Full Training

Run the complete pipeline:

```bash
python main.py
```

**What happens:**

1. **Data Loading** (~2-3 minutes)
   - Loads all 5,915 data files
   - Pads/truncates to 5,500 timepoints
   
2. **Filtering** (~1 minute)
   - Groups by sentence
   - Keeps sentences with ≥3 samples
   - Typical result: ~400-500 unique sentences
   
3. **Train/Test Split** (~1 minute)
   - 80% train, 20% test per sentence
   - Typical: ~3,500 train, ~900 test samples
   
4. **Data Augmentation** (~5-10 minutes)
   - Creates 2 augmented versions per sample
   - Applies: amplitude scaling, noise, time shift
   - Typical: ~10,000 augmented training samples
   
5. **CNN Training** (~15-20 minutes)
   - 3 epochs over augmented data
   - Batch size: 8
   - Saves checkpoint after training
   
6. **Feature Extraction** (~5 minutes)
   - Extracts features for all train/test samples
   - Output: 32-dimensional features per timepoint
   
7. **HMM Training** (~10-15 minutes)
   - Trains one HMM per unique sentence
   - 10 Baum-Welch iterations per HMM
   - Progress shown for each sentence
   
8. **Evaluation** (~1-2 minutes)
   - Predicts on test set
   - Calculates accuracy metrics

**Total time**: 30-60 minutes

### Step 3: Monitor Progress

Watch for these indicators:

✅ **Good signs:**
```
✓ Loaded 5915 sequences
✓ Found 472 sentences with >= 3 samples
Epoch [1/3], Train Loss: 0.0234
✓ Successfully trained 472 models
```

⚠️ **Warning signs:**
```
✗ No data loaded
✗ No sentences found with enough samples
✗ Failed to train 50 models
```

---

## Testing and Evaluation

### Automatic Evaluation

The pipeline automatically evaluates on the test set and prints:

1. **Sample Predictions** (first 10)
2. **Per-Sentence Accuracy** (all sentences)
3. **Overall Accuracy**

### Understanding the Output

#### Sample Predictions

```
Sample 1:
  True: Henry Ford, with his son Edsel, founded the Ford Foundation...
  Pred: Henry Ford, with his son Edsel, founded the Ford Foundation... (Score: -1234.56)
  Result: ✓ CORRECT
```

- **True**: Ground truth sentence
- **Pred**: Predicted sentence
- **Score**: Log-likelihood from HMM (higher is better)
- **Result**: Whether prediction matches truth

#### Per-Sentence Accuracy

```
Per-Sentence Accuracy (472 unique sentences):
----------------------------------------------------------------------
  20.0% (1/5) - Henry Ford, with his son Edsel, founded the...
  33.3% (2/6) - After this initial success, Ford left Edison...
  50.0% (3/6) - With his interest in race cars, he formed a...
  ...
```

Shows accuracy for each sentence individually. Sorted by accuracy (lowest first) to identify problem sentences.

#### Overall Metrics

```
Overall Accuracy: 312/864 (36.11%)
```

Total correct predictions across all test samples.

---

## Understanding Results

### What is Good Accuracy?

Accuracy depends on the number of unique sentences:

| Unique Sentences | Expected Accuracy | Interpretation |
|-----------------|-------------------|----------------|
| 10-50 | 40-70% | Good performance |
| 50-200 | 30-50% | Reasonable |
| 200-500 | 20-40% | Expected for large vocabulary |
| 500+ | 15-30% | Challenging task |

**Random baseline**: 1 / (number of unique sentences)
- Example: 472 sentences → random = 0.21% → 36% is **171x better than random**

### Factors Affecting Accuracy

1. **Number of unique sentences**: More sentences = harder task
2. **Samples per sentence**: More samples = better HMM training
3. **EEG signal quality**: Cleaner signals = better features
4. **Sentence similarity**: Similar sentences are harder to distinguish
5. **CNN feature quality**: Better features = better HMM performance

### Analyzing Results

#### High Accuracy Sentences

Sentences with >50% accuracy typically have:
- More training samples (5-10+)
- Distinct EEG patterns
- Less similarity to other sentences

#### Low Accuracy Sentences

Sentences with <20% accuracy may have:
- Few training samples (3-4)
- Similar content to other sentences
- Noisy EEG signals

**Action**: Consider removing low-sample sentences or collecting more data.

---

## Advanced Usage

### Custom Training Configuration

```bash
# More CNN training (may improve features)
python main.py --cnn-epochs 5

# More complex HMMs (may improve classification)
python main.py --hmm-states 5

# More data augmentation (may improve generalization)
python main.py --num-aug 5

# Stricter filtering (better HMM training, fewer sentences)
python main.py --min-samples 5

# Combine multiple options
python main.py --cnn-epochs 5 --hmm-states 5 --num-aug 5
```

### Resume from Checkpoint

If training is interrupted:

```bash
python main.py --resume checkpoints/cnn_encoder.pth
```

This loads the saved CNN encoder and continues from there.

### Batch Processing

To process multiple configurations:

```bash
# Create a script
for states in 3 4 5; do
    python main.py --hmm-states $states --save-models
    mv checkpoints/hmm_models.pkl checkpoints/hmm_models_${states}states.pkl
done
```

---

## Tips for Improving Performance

### 1. Increase CNN Training

```bash
python main.py --cnn-epochs 10
```

**Why**: Better feature extraction → better HMM performance
**Trade-off**: Longer training time

### 2. More Data Augmentation

```bash
python main.py --num-aug 5
```

**Why**: More training data → better generalization
**Trade-off**: Longer training time, more memory

### 3. Tune HMM Complexity

```bash
# Try different state counts
python main.py --hmm-states 4
python main.py --hmm-states 5
```

**Why**: Optimal complexity depends on data
**Trade-off**: More states = more parameters to learn

### 4. Filter Low-Sample Sentences

```bash
python main.py --min-samples 5
```

**Why**: Better HMM training with more samples
**Trade-off**: Fewer unique sentences in the model

### 5. Adjust Train/Test Split

```bash
python main.py --train-split 0.9
```

**Why**: More training data → better models
**Trade-off**: Smaller test set for evaluation

### 6. Experiment with Learning Rate

Edit `src/config.py`:
```python
CNN_LEARNING_RATE = 5e-4  # Lower learning rate
```

**Why**: May help CNN converge better
**Trade-off**: May need more epochs

---

## Troubleshooting Training Issues

### Issue: "No sentences found with enough samples"

**Cause**: Not enough samples per sentence

**Solutions:**
```bash
# Lower the minimum requirement
python main.py --min-samples 2

# Or check your data
python -c "from src.data_loader import DataLoader; loader = DataLoader('processed_data'); loader.load_mapping(); print(f'Total files: {len(loader.get_all_files())}')"
```

### Issue: Memory errors during CNN training

**Cause**: Too much data in memory

**Solutions:**
```bash
# Reduce batch size
python main.py --cnn-batch-size 4

# Or use quick test mode
python main.py --quick-test

# Or reduce augmentation
python main.py --num-aug 1
```

### Issue: HMM training failures

**Cause**: Numerical instability in Baum-Welch

**Solutions:**
- Check for NaN values in data
- Reduce HMM states: `--hmm-states 2`
- Ensure enough samples per sentence

### Issue: Very low accuracy (<10%)

**Possible causes:**
1. Too many unique sentences (>500)
2. Poor EEG signal quality
3. Insufficient training data per sentence

**Solutions:**
- Increase `--min-samples` to filter out low-sample sentences
- Increase `--cnn-epochs` for better features
- Check data quality and preprocessing

---

## Expected Training Output

### Successful Run

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
  (Total unique sentences: 1247)

STEP 3: Creating Train/Test Split
----------------------------------------------------------------------
✓ Training Set: 3456 samples
✓ Test Set: 864 samples

STEP 4: Augmenting Training Data
----------------------------------------------------------------------
✓ Total training samples after augmentation: 10368
  (Augmentation factor: 3.0x)

STEP 5: Training CNN Autoencoder
----------------------------------------------------------------------
Starting Autoencoder training...
Device: cpu, Epochs: 3, Learning Rate: 0.001
----------------------------------------------------------------------
Epoch [1/3], Train Loss: 0.0234
Epoch [2/3], Train Loss: 0.0187
Epoch [3/3], Train Loss: 0.0156
----------------------------------------------------------------------
✓ Model checkpoint saved to checkpoints/cnn_encoder.pth

STEP 6: Extracting Features for HMM
----------------------------------------------------------------------
Extracting training features...
✓ Extracted 10368 training feature sequences
Extracting test features...
✓ Extracted 864 test feature sequences

STEP 7: Training HMM Sentence Predictor
----------------------------------------------------------------------

Training 472 distinct sentence models...
----------------------------------------------------------------------
[1/472] Training: 'Henry Ford, with his son Edsel, founded the...' (8 samples)
Iteration 1/10, Log-Likelihood: -2345.6789
...
----------------------------------------------------------------------
✓ Successfully trained 472 models
✓ Saved 472 HMM models to checkpoints/hmm_models.pkl

STEP 8: Evaluating on Test Set
----------------------------------------------------------------------
Running predictions on 864 test samples...

Sample 1:
  True: Henry Ford, with his son Edsel, founded the Ford Foundation...
  Pred: Henry Ford, with his son Edsel, founded the Ford Foundation... (Score: -1234.56)
  Result: ✓ CORRECT

...

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

---

## Next Steps After Training

1. **Analyze results**: Review per-sentence accuracy to identify patterns
2. **Tune hyperparameters**: Try different configurations based on results
3. **Collect more data**: Focus on low-accuracy sentences
4. **Experiment with architecture**: Modify CNN or HMM parameters in code
5. **Deploy model**: Use saved checkpoints for inference on new data

For questions or issues, refer to the main README.md or check the implementation plan.
