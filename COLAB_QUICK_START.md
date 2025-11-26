# Google Colab Quick Start Guide

Complete guide to run the improved EEG-to-Text pipeline on Google Colab with your data on Google Drive.

## 📋 Prerequisites

- Google account
- Your dataset ready (5,915 CSV files + sentence_mapping.csv)
- ~2-3 GB of Google Drive storage

## 🚀 Step-by-Step Instructions

### Step 1: Upload Your Data to Google Drive

1. **Create a folder in Google Drive:**
   - Go to https://drive.google.com
   - Click "New" → "New folder"
   - Name it `ML_Project` (or any name you prefer)

2. **Upload your project files:**

   Upload these folders/files to your `ML_Project` folder:
   ```
   ML_Project/
   ├── processed_data/          ← Upload this entire folder
   │   ├── rawdata_0001.csv
   │   ├── rawdata_0002.csv
   │   ├── ... (all 5,915 files)
   │   └── sentence_mapping.csv
   ├── src/                     ← Upload this entire folder
   │   ├── config.py
   │   ├── data_loader.py
   │   ├── feature_extractor.py
   │   ├── hmm_model.py
   │   ├── predictor.py
   │   └── utils.py
   ├── main.py                  ← Upload this file
   └── main_memory_efficient.py ← Upload this file (optional)
   ```

3. **Upload method:**
   - **Option A (Recommended):** Use Google Drive Desktop app to sync the folder
   - **Option B:** Drag and drop folders directly in browser
   - **Expected upload time:** 10-30 minutes depending on internet speed

### Step 2: Upload Notebook to Google Drive

1. Upload `EEG_to_Text_Colab_Improved.ipynb` to Google Drive
2. Right-click the notebook → "Open with" → "Google Colaboratory"
3. If you don't see "Google Colaboratory", click "Connect more apps" and install it

### Step 3: Enable GPU in Colab

1. In the Colab notebook, go to **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Set **GPU type** to **T4** (free tier)
4. Click **Save**

**Why GPU?** CNN training will be 5-10x faster (~5 min vs ~30 min)

### Step 4: Update the Data Path

In the notebook, find Cell 3 and update the path:

```python
# UPDATE THIS PATH to match your Google Drive folder
DRIVE_PATH = '/content/drive/MyDrive/ML_Project'
```

**How to find your path:**
1. In Google Drive, navigate to your uploaded folder
2. The path structure is always: `/content/drive/MyDrive/YOUR_FOLDER_NAME`
3. Example: If your folder is `ML_Project`, use `/content/drive/MyDrive/ML_Project`

### Step 5: Run the Notebook

1. Click **Runtime** → **Run all**
2. Or run cells one by one from top to bottom
3. When prompted, authorize Google Drive access

**Expected flow:**
- Cell 1: Mounts Google Drive (requires authorization)
- Cell 2: Installs dependencies (~30 seconds)
- Cell 3: Changes to your data directory
- Cell 4: Configures GPU
- Cell 5: Verifies dataset
- Cell 6: Optional quick test (~2-3 min)
- Cell 7: **Full training (~30-60 min)** ⭐
- Cells 8-11: Results and inference

### Step 6: Monitor Progress

During training, you'll see:

```
======================================================================
EEG-TO-TEXT HMM PIPELINE
======================================================================

STEP 5: Training CNN Encoder (Supervised)
----------------------------------------------------------------------
Epoch [1/5], Train Loss: 3.2456, Train Acc: 25.34%
Epoch [2/5], Train Loss: 1.8234, Train Acc: 52.11%
Epoch [3/5], Train Loss: 1.2145, Train Acc: 68.45%
Epoch [4/5], Train Loss: 0.8932, Train Acc: 78.92%
Epoch [5/5], Train Loss: 0.6734, Train Acc: 84.56%
✓ Best training accuracy: 84.56%

STEP 6: Extracting Features for HMM + Normalization
----------------------------------------------------------------------
✓ Features normalized (mean=0, std=1)

STEP 7: Training HMM Sentence Predictor
----------------------------------------------------------------------
✓ Successfully trained 344 models

STEP 8: Evaluating on Test Set
----------------------------------------------------------------------
Overall Accuracy: 620/1000 (62.00%)

======================================================================
PIPELINE COMPLETED
======================================================================
Final Accuracy: 62.00%
```

**Good signs:**
- ✅ CNN training accuracy reaches 70-90%
- ✅ All 344 HMM models train successfully
- ✅ Final accuracy is 50-70%

### Step 7: Results

Models are **automatically saved to your Google Drive** at:
```
/ML_Project/checkpoints/
├── cnn_encoder.pth    (~50-100 MB)
└── hmm_models.pkl     (~10-50 MB)
```

You can:
- Download models using Cell 10
- They'll persist in Google Drive even after session ends
- Use them for inference later

## ⏱️ Expected Timeline

| Step | Time | Notes |
|------|------|-------|
| Upload to Drive | 10-30 min | One-time setup |
| Install dependencies | 30 sec | Per session |
| Quick test | 2-3 min | Optional verification |
| Full training | 30-60 min | With GPU |
| Total first run | 45-90 min | Including setup |

## 💾 Resource Requirements

**Free Colab Tier:**
- RAM: 12-15 GB ✓ (Sufficient)
- GPU: T4 with 16 GB VRAM ✓ (Perfect)
- Session: 12 hours max ✓ (More than enough)
- Storage: 0 (saves to your Drive)

**Your dataset:**
- 5,915 files × ~50 KB = ~300 MB
- With augmentation in memory: ~12 GB (fits comfortably)

## 🔧 Troubleshooting

### Issue: "No such file or directory"

**Cause:** Wrong `DRIVE_PATH`

**Solution:**
1. In Google Drive, navigate to your folder
2. Copy the exact folder name (case-sensitive!)
3. Update path: `/content/drive/MyDrive/YOUR_EXACT_FOLDER_NAME`

Example:
```python
# ✗ Wrong
DRIVE_PATH = '/content/drive/MyDrive/ml_project'  # lowercase

# ✓ Correct
DRIVE_PATH = '/content/drive/MyDrive/ML_Project'  # matches Drive
```

### Issue: "Out of Memory"

**Solution 1:** Use memory-efficient version
```python
# In Cell 8 instead of Cell 7
!python main_memory_efficient.py --num-aug 2 --save-models
```

**Solution 2:** Reduce augmentation
```python
!python main.py --num-aug 1 --save-models  # Only 1x augmentation
```

**Solution 3:** Limit files
```python
!python main.py --max-files 3000 --num-aug 2  # Use fewer files
```

### Issue: "GPU not available"

**Solution:**
1. Runtime → Change runtime type
2. Set Hardware accelerator to **GPU**
3. Click Save
4. Run this to verify:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   ```

### Issue: "Session disconnected"

**Don't worry!** Models are saved to Google Drive automatically.

**To resume:**
```python
!python main.py --resume checkpoints/cnn_encoder.pth --num-aug 2
```

### Issue: Low accuracy (<40%)

**Check:**
1. Is GPU enabled? (Should see "GPU Available: True")
2. Did CNN training reach 70-90% accuracy?
3. Are you using the improved notebook (`EEG_to_Text_Colab_Improved.ipynb`)?

**Try:**
```python
# More CNN training
!python main.py --cnn-epochs 10 --num-aug 2

# More HMM complexity
!python main.py --hmm-states 6 --num-aug 2
```

## 📊 Expected Results

### With Improvements (Current Version)

- **CNN Training Accuracy**: 70-90%
- **Final Test Accuracy**: 50-70%
- **vs Random (0.29%)**: 172-241x better
- **vs Baseline (36%)**: +14-34% improvement

### Key Improvements in This Version

✅ **Supervised CNN** - Classification loss (not reconstruction)
✅ **Diagonal Covariance** - More stable with limited data
✅ **Feature Normalization** - Better HMM convergence
✅ **Enhanced Augmentation** - 6 realistic techniques
✅ **Better Hyperparameters** - 5 states, 5 epochs

## 💡 Tips for Best Results

1. **Always enable GPU** - 5-10x faster training
2. **Keep browser tab open** - Prevents session timeout
3. **Check GPU usage**: Runtime → View resources
4. **Close other Colab notebooks** - Frees up resources
5. **Download models** - Save to local machine as backup

## 📱 Running from Phone/Tablet

Yes, you can! Colab works on mobile browsers:
1. Upload data via Google Drive app
2. Open notebook in Chrome/Safari
3. Training runs in cloud (doesn't use your device's resources)
4. May be harder to monitor, but works fine

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Read error messages carefully
3. Verify all files uploaded correctly
4. Make sure GPU is enabled
5. Try the quick test first (Cell 6)

## 🎯 Quick Checklist

Before running:
- [ ] Data uploaded to Google Drive
- [ ] Notebook opened in Colab
- [ ] GPU enabled (Runtime → Change runtime type)
- [ ] DRIVE_PATH updated in Cell 3
- [ ] All required files verified (Cell 3 output)

During training:
- [ ] CNN accuracy increasing (should reach 70-90%)
- [ ] No memory errors
- [ ] HMM models training successfully

After training:
- [ ] Models saved to checkpoints/
- [ ] Final accuracy 50-70% (or at least >45%)
- [ ] Models backed up (optional)

---

## 🚀 You're Ready!

Once data is uploaded and path is updated, just click **Runtime → Run all** and wait 30-60 minutes.

Models will be automatically saved to your Google Drive and you'll see accuracy results at the end!

Good luck! 🎉
