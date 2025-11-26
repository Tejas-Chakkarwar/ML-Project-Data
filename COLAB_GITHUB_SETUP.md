# Google Colab Setup Guide - Using GitHub Repository

## 🚀 Quick Start (5 minutes)

This guide shows you how to run the EEG-to-Text pipeline on Google Colab using code from GitHub and data from Google Drive.

---

## 📋 Prerequisites

1. **Google Account** (for Colab and Drive)
2. **Dataset uploaded to Google Drive** (~35 GB)
3. **GitHub repository** (already set up)

---

## 🎯 Step-by-Step Instructions

### Step 1: Upload Your Dataset to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create a folder named `ML_Project_Data`
3. Inside it, create a folder named `processed_data`
4. Upload all your EEG CSV files to `processed_data/`:
   - All `rawdata_*.csv` files (5,915 files)
   - `sentence_mapping.csv`
   - `sentence_mapping.json`

**Final structure:**
```
Google Drive/
└── ML_Project_Data/
    └── processed_data/
        ├── rawdata_0001.csv
        ├── rawdata_0002.csv
        ├── ... (all 5,915 files)
        ├── sentence_mapping.csv
        └── sentence_mapping.json
```

**Upload time:** 10-30 minutes (one-time setup)

---

### Step 2: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → New notebook**
3. Enable GPU:
   - **Runtime → Change runtime type**
   - Set **Hardware accelerator** to **GPU**
   - Set **GPU type** to **T4**
   - Click **Save**

---

### Step 3: Run Setup Cells in Colab

Copy and paste these cells into your Colab notebook:

#### Cell 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Action required:** Authorize access when prompted

---

#### Cell 2: Clone GitHub Repository

```python
!git clone https://github.com/Tejas-Chakkarwar/ML-Project-Data.git
%cd ML-Project-Data
!ls -la
```

**Expected output:** List of project files (main.py, src/, etc.)

---

#### Cell 3: Install Dependencies

```python
!pip install -q torch numpy pandas
```

**Time:** ~30 seconds

---

#### Cell 4: Link Google Drive Data

```python
import os

# UPDATE THIS PATH if your Drive folder has a different name
DRIVE_DATA_PATH = '/content/drive/MyDrive/ML_Project_Data/processed_data'

# Create symbolic link to your data
if os.path.exists('processed_data'):
    !rm -rf processed_data
!ln -s "{DRIVE_DATA_PATH}" processed_data

# Verify data is accessible
!ls -lh processed_data/ | head -10
!wc -l processed_data/sentence_mapping.csv
```

**Expected output:** 
- List of CSV files
- `5916 processed_data/sentence_mapping.csv`

---

#### Cell 5: Configure for GPU

```python
# Update config to use GPU
!sed -i "s/CNN_DEVICE = 'cpu'/CNN_DEVICE = 'cuda'/" src/config.py

# Verify GPU is available
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

**Expected output:** 
```
GPU Available: True
GPU Name: Tesla T4
```

---

#### Cell 6: Quick Test (Optional - 2-3 minutes)

```python
!python main.py --quick-test
```

**Expected:** 30-40% accuracy with 100 files

---

#### Cell 7: Full Training (30-60 minutes) ⭐

```python
# Recommended settings for best results
!python main.py \
  --cnn-epochs 5 \
  --hmm-states 5 \
  --num-aug 2 \
  --save-models \
  --verbose
```

**Expected:** 55-65% accuracy

---

#### Cell 8: Save Models to Google Drive

```python
# Copy trained models to Google Drive for persistence
!mkdir -p /content/drive/MyDrive/ML_Project_Data/checkpoints
!cp -r checkpoints/* /content/drive/MyDrive/ML_Project_Data/checkpoints/

print("✓ Models saved to Google Drive!")
print("Location: /MyDrive/ML_Project_Data/checkpoints/")
```

---

#### Cell 9: View Results

```python
# Display final results
!tail -50 /content/ML-Project-Data/training.log 2>/dev/null || echo "Check output above for results"
```

---

### Step 4: Advanced Options

#### Memory-Efficient Version (if OOM errors)

```python
!python main_memory_efficient.py \
  --num-aug 1 \
  --cnn-batch-size 4 \
  --save-models
```

#### Optimized for Best Accuracy

```python
!python main.py \
  --cnn-epochs 10 \
  --hmm-states 7 \
  --num-aug 3 \
  --cnn-batch-size 16 \
  --save-models
```

**Expected:** 60-70% accuracy (takes 60-90 min)

#### Resume from Checkpoint

```python
!python main.py \
  --resume checkpoints/cnn_encoder.pth \
  --num-aug 2 \
  --save-models
```

---

## 🔧 Troubleshooting

### Issue: "No such file or directory: processed_data"

**Solution:** Check your Drive path in Cell 4
```python
# List your Drive folders
!ls -la /content/drive/MyDrive/

# Update DRIVE_DATA_PATH to match your folder name
DRIVE_DATA_PATH = '/content/drive/MyDrive/YOUR_FOLDER_NAME/processed_data'
```

### Issue: "GPU not available"

**Solution:**
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. Save and reconnect

### Issue: "Out of Memory"

**Solution:** Use memory-efficient version
```python
!python main_memory_efficient.py --num-aug 1
```

### Issue: Session disconnected

**Don't worry!** Models are saved to Google Drive.

**To resume:**
```python
!python main.py --resume /content/drive/MyDrive/ML_Project_Data/checkpoints/cnn_encoder.pth
```

---

## 📊 Expected Performance

| Configuration | Accuracy | Time (GPU) |
|--------------|----------|------------|
| Quick test | 30-40% | 2-3 min |
| Default | 55-65% | 30-45 min |
| Optimized | 60-70% | 60-90 min |

---

## 💡 Tips for Best Results

1. ✅ **Always enable GPU** - 5-10x faster
2. ✅ **Keep browser tab open** - Prevents disconnection
3. ✅ **Save models to Drive** - Persist after session ends
4. ✅ **Start with quick test** - Verify everything works
5. ✅ **Monitor GPU usage** - Runtime → View resources

---

## 🎯 Complete Workflow Summary

```bash
# 1. One-time setup (10-30 min)
- Upload data to Google Drive
- Create Colab notebook

# 2. Every session (30-60 min)
- Mount Drive
- Clone GitHub repo
- Install dependencies
- Link data
- Run training
- Save models to Drive
```

---

## 📝 Full Colab Notebook Template

Here's a complete ready-to-use notebook:

```python
# ========== CELL 1: Setup ==========
from google.colab import drive
drive.mount('/content/drive')

# ========== CELL 2: Clone Repo ==========
!git clone https://github.com/Tejas-Chakkarwar/ML-Project-Data.git
%cd ML-Project-Data

# ========== CELL 3: Install ==========
!pip install -q torch numpy pandas

# ========== CELL 4: Link Data ==========
import os
DRIVE_DATA_PATH = '/content/drive/MyDrive/ML_Project_Data/processed_data'
if os.path.exists('processed_data'):
    !rm -rf processed_data
!ln -s "{DRIVE_DATA_PATH}" processed_data
!wc -l processed_data/sentence_mapping.csv

# ========== CELL 5: Enable GPU ==========
!sed -i "s/CNN_DEVICE = 'cpu'/CNN_DEVICE = 'cuda'/" src/config.py
import torch
print(f"GPU: {torch.cuda.is_available()}")

# ========== CELL 6: Train ==========
!python main.py --cnn-epochs 5 --hmm-states 5 --num-aug 2 --save-models

# ========== CELL 7: Save to Drive ==========
!mkdir -p /content/drive/MyDrive/ML_Project_Data/checkpoints
!cp -r checkpoints/* /content/drive/MyDrive/ML_Project_Data/checkpoints/
print("✓ Done!")
```

---

## 🎉 You're Ready!

Just copy the cells above into a new Colab notebook and run them in order. The entire process takes 30-60 minutes including training.

**Questions?** Check the troubleshooting section above or refer to the main README.md.

Good luck! 🚀
