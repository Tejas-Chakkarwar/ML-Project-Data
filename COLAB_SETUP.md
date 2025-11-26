# Running on Google Colab

Complete guide for running the EEG-to-Text pipeline on Google Colab with GPU acceleration.

## Why Google Colab?

✅ **Free GPU access** - Much faster CNN training (5-10x speedup)
✅ **More RAM** - 12-15 GB vs your local machine
✅ **No local setup** - Everything runs in the cloud
✅ **Can run full dataset** - With 2x augmentation

## Setup Steps

### 1. Upload Dataset to Google Drive

1. Create a folder in Google Drive (e.g., "ML Project Data")
2. Upload these folders/files:
   - `processed_data/` (all CSV files)
   - `src/` (all Python files)
   - `main.py`
   - `main_memory_efficient.py`

**Upload method:**
- Use Google Drive web interface
- Or use [Google Drive Desktop](https://www.google.com/drive/download/) to sync

**Expected upload time:** 10-30 minutes (depending on internet speed)

### 2. Open the Colab Notebook

1. Upload `EEG_to_Text_Colab.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory
3. If you don't see "Google Colaboratory", click "Connect more apps" and install it

### 3. Enable GPU

1. In Colab: **Runtime** → **Change runtime type**
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (free tier)
4. Click **Save**

### 4. Run the Notebook

Run each cell in order:

1. **Mount Google Drive** - Authorize access
2. **Install Dependencies** - Check GPU availability
3. **Set Up Directory** - Update `DRIVE_PATH` to your folder
4. **Update Config** - Enable GPU for CNN
5. **Check Data** - Verify files loaded correctly
6. **Run Training** - Full pipeline with 2x augmentation

### 5. Monitor Progress

The training will show:
```
======================================================================
EEG-TO-TEXT HMM PIPELINE
======================================================================

STEP 1: Loading Data
----------------------------------------------------------------------
✓ Loaded 5915 sequences

STEP 2: Filtering for Cross-Subject Training
----------------------------------------------------------------------
✓ Found 344 sentences with >= 3 samples

...
```

**Expected time:** 30-45 minutes total

### 6. Download Results

After training completes:
- Models are saved to your Google Drive
- Optionally download using the notebook's download cell
- Results include:
  - `checkpoints/cnn_encoder.pth` (~50-100 MB)
  - `checkpoints/hmm_models.pkl` (~10-50 MB)

## Configuration Options

### Full Dataset (Recommended)

```python
!python main.py --num-aug 2 --save-models
```

**Specs:**
- 5,915 files
- 2x augmentation
- ~344 unique sentences
- ~12-15 GB RAM needed
- ~30-45 minutes

### Memory-Efficient Version

If you get memory errors:

```python
!python main_memory_efficient.py --num-aug 2 --save-models
```

**Benefits:**
- Processes data in batches
- Lower memory footprint
- Slightly slower but more reliable

### Custom Options

```python
# Fewer files for faster testing
!python main.py --max-files 2000 --num-aug 2

# More CNN training
!python main.py --num-aug 2 --cnn-epochs 5

# More complex HMMs
!python main.py --num-aug 2 --hmm-states 5
```

## Expected Results

### Performance Metrics

With full dataset (5,915 files, 344 sentences):

- **Accuracy**: 25-40%
- **Random baseline**: 0.29% (1/344)
- **Improvement over random**: 86-138x better

### GPU Speedup

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| CNN Training | 15-20 min | 2-3 min | 5-7x |
| Feature Extraction | 5 min | 1 min | 5x |
| HMM Training | 10-15 min | 10-15 min | 1x (CPU only) |
| **Total** | **45-60 min** | **30-45 min** | **1.5-2x** |

### Memory Usage

| Stage | RAM Used | Peak RAM |
|-------|----------|----------|
| Data Loading | 3-4 GB | 4 GB |
| Augmentation | 8-10 GB | 12 GB |
| CNN Training | 6-8 GB | 10 GB |
| HMM Training | 4-6 GB | 8 GB |

## Troubleshooting

### Issue: "Out of Memory"

**Solution 1:** Use memory-efficient version
```python
!python main_memory_efficient.py --num-aug 2
```

**Solution 2:** Reduce augmentation
```python
!python main.py --num-aug 1
```

**Solution 3:** Limit files
```python
!python main.py --max-files 3000 --num-aug 2
```

### Issue: "GPU not available"

**Check:**
1. Runtime → Change runtime type → GPU is selected
2. Run the GPU check cell:
```python
import torch
print(torch.cuda.is_available())
```

**If still not available:**
- Free tier GPU may be temporarily unavailable
- Try again later or use CPU (will be slower)

### Issue: "Session timeout"

**Prevention:**
- Keep the browser tab open
- Interact with the notebook occasionally

**Recovery:**
- Models are saved to Google Drive
- Can resume from checkpoint:
```python
!python main.py --resume checkpoints/cnn_encoder.pth --num-aug 2
```

### Issue: "Drive path not found"

**Fix:**
Update the `DRIVE_PATH` in cell 3:
```python
DRIVE_PATH = '/content/drive/MyDrive/YOUR_FOLDER_NAME'
```

Make sure to use the exact folder name from your Google Drive.

## Tips for Best Results

### 1. Use GPU Runtime

Always enable GPU for 5-10x faster CNN training.

### 2. Close Other Tabs

Free up browser memory by closing unnecessary tabs.

### 3. Monitor RAM Usage

In Colab, check: **Runtime** → **View resources**

If RAM usage is high (>90%), consider:
- Using memory-efficient version
- Reducing augmentation
- Limiting files

### 4. Save Incrementally

Models are saved automatically to Google Drive after each stage.

### 5. Download Results

Download trained models to use locally:
```python
from google.colab import files
files.download('checkpoints/cnn_encoder.pth')
files.download('checkpoints/hmm_models.pkl')
```

## Colab Limitations

### Free Tier

- **RAM**: 12-15 GB
- **GPU**: T4 (16 GB VRAM)
- **Session**: 12 hours max
- **Daily limit**: ~12 hours of GPU usage

### Colab Pro ($10/month)

- **RAM**: Up to 50 GB
- **GPU**: Better GPUs (V100, A100)
- **Session**: 24 hours max
- **Priority access**: No daily limits

For this project, **free tier is sufficient**.

## After Training

### Use Trained Models

1. Models are saved in `checkpoints/` on Google Drive
2. Download to local machine
3. Use for inference:

```python
from src.feature_extractor import CNNEEGEncoder
from src.predictor import SentencePredictor

# Load models
encoder = CNNEEGEncoder(...)
encoder.load_state_dict(torch.load('checkpoints/cnn_encoder.pth'))

predictor = SentencePredictor(...)
predictor.load('checkpoints/hmm_models.pkl')

# Run inference
# ... (see notebook cell 10)
```

### Analyze Results

The training output includes:
- Overall accuracy
- Per-sentence accuracy
- Sample predictions
- Confusion patterns

Review these to understand:
- Which sentences are easy/hard to classify
- Where the model makes mistakes
- Opportunities for improvement

## Next Steps

1. **Upload dataset to Google Drive** (~10-30 min)
2. **Open Colab notebook** and enable GPU
3. **Run training** with 2x augmentation (~30-45 min)
4. **Download models** and analyze results
5. **Experiment** with different hyperparameters if needed

Good luck! 🚀
