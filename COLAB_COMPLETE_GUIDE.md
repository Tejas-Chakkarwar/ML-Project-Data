# 🎯 Complete Colab Setup - README

## 📌 Overview

Your ML project is now fully configured to run on Google Colab! This setup allows you to:
- ✅ Access code from GitHub
- ✅ Access dataset from Google Drive  
- ✅ Train on free GPU (Tesla T4)
- ✅ Save models back to Drive
- ✅ Resume training if disconnected

---

## 📊 Workflow Diagram

![Colab Workflow](/Users/tejaschakkarwar/.gemini/antigravity/brain/df5371ae-a10f-407b-8221-28eb645c1f57/colab_workflow_diagram_1764120520339.png)

---

## 🚀 Quick Start

### ⚡ I Want to Start Immediately
1. Read this guide (5 min)
2. Upload **`EEG_to_Text_Colab_Updated.ipynb`** to Colab
3. Run all cells!

### 📚 I Want More Details
1. Read the complete instructions below
2. Check the **Troubleshooting** section if needed
3. Upload **`EEG_to_Text_Colab_Updated.ipynb`** to Colab
4. Follow the notebook step-by-step

---

## 📁 Essential Files

### 🎯 For Running on Colab

| File | Purpose |
|------|---------|
| **`EEG_to_Text_Colab_Updated.ipynb`** | ⭐ Main Colab notebook - Upload this to Colab |
| **`COLAB_COMPLETE_GUIDE.md`** | Complete setup guide with all instructions |
| **`README.md`** | Project overview and documentation |

### 💻 Code Files (Auto-loaded from GitHub)

| File | Purpose |
|------|---------|
| `main.py` | Main training script |
| `main_memory_efficient.py` | Memory-optimized version |
| `src/` | Source code modules |
| `requirements.txt` | Python dependencies |

---

## 🔗 Important Links

### Your Resources

| Resource | Link |
|----------|------|
| **GitHub Repo (Code)** | https://github.com/Tejas-Chakkarwar/ML-Project-Data.git |
| **Google Drive (Dataset)** | https://drive.google.com/drive/folders/1R3RAoh-G6Wa3jwpN3_qkYPGcf87K8Lbw?usp=sharing |
| **Google Colab** | https://colab.research.google.com |

### What's Where

```
GitHub Repository
├── main.py                    ← Training script
├── main_memory_efficient.py   ← Memory-optimized version
├── src/                       ← Source code
│   ├── config.py
│   ├── data_loader.py
│   ├── cnn_encoder.py
│   ├── hmm.py
│   └── ...
└── requirements.txt           ← Dependencies

Google Drive
└── ML_Project_Data/
    ├── processed_data/        ← Your dataset
    │   ├── rawdata_*.csv      ← EEG data files
    │   ├── sentence_mapping.csv
    │   └── sentence_mapping.json
    └── checkpoints/           ← Saved models (after training)
        ├── cnn_encoder.pth
        └── hmm_models.pkl
```

---

## ⚡ Super Quick Start (TL;DR)

```bash
# 1. Add shared Drive folder to "My Drive"
#    https://drive.google.com/drive/folders/1R3RAoh-G6Wa3jwpN3_qkYPGcf87K8Lbw

# 2. Open Colab, enable GPU (Runtime → Change runtime type → GPU)

# 3. Run these cells:
```

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone code
!git clone https://github.com/Tejas-Chakkarwar/ML-Project-Data.git
%cd ML-Project-Data

# Install
!pip install -q torch numpy pandas

# Link data (UPDATE PATH!)
!ln -s "/content/drive/MyDrive/ML_Project_Data/processed_data" processed_data

# Enable GPU
!sed -i "s/CNN_DEVICE = 'cpu'/CNN_DEVICE = 'cuda'/" src/config.py

# Train!
!python main.py --cnn-epochs 5 --hmm-states 5 --num-aug 2 --save-models

# Save to Drive
!mkdir -p /content/drive/MyDrive/ML_Project_Data/checkpoints
!cp -r checkpoints/* /content/drive/MyDrive/ML_Project_Data/checkpoints/
```

---

## ✅ Setup Checklist

### Before You Start

- [ ] Google account ready
- [ ] Shared Drive folder added to "My Drive"
- [ ] Folder renamed to `ML_Project_Data` (optional)
- [ ] Verified `processed_data` subfolder exists
- [ ] CSV files and mapping files present

### In Colab

- [ ] Uploaded `EEG_to_Text_Colab_Updated.ipynb`
- [ ] GPU enabled (Runtime → Change runtime type)
- [ ] Mounted Google Drive
- [ ] Found correct data path
- [ ] Updated `DRIVE_DATA_PATH` in notebook

### During Training

- [ ] Quick test passed (30-40% accuracy)
- [ ] GPU showing as available
- [ ] Training running without errors
- [ ] Browser tab kept open

### After Training

- [ ] Models saved to Drive
- [ ] Accuracy 55-65% achieved
- [ ] Checkpoints backed up

---

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| **Setup Time** | 5 minutes (one-time) |
| **Training Time** | 30-60 minutes |
| **Expected Accuracy** | 55-65% |
| **Random Baseline** | 0.29% |
| **Improvement** | **172-241x better!** |

---

## 🔧 Common Issues & Solutions

### "Path not found"
```python
!find /content/drive/MyDrive -type d -name "processed_data"
# Use the output to update DRIVE_DATA_PATH
```

### "GPU not available"
- Runtime → Change runtime type → GPU → T4 → Save

### "Out of memory"
```python
!python main_memory_efficient.py --num-aug 1 --save-models
```

### "Session disconnected"
```python
# Resume from checkpoint
!python main.py --resume /content/drive/MyDrive/ML_Project_Data/checkpoints/cnn_encoder.pth
```

---

## 💡 Pro Tips

1. **Start with quick test** - Verifies everything works (2-3 min)
2. **Keep tab open** - Prevents disconnection
3. **Monitor GPU** - Runtime → View resources
4. **Save to Drive** - Models persist after session
5. **Use shortcuts** - Don't count against storage quota

---

## 🎯 Success Indicators

You'll know it's working when:

1. ✅ `GPU Available: True`
2. ✅ `wc -l processed_data/sentence_mapping.csv` shows ~5916
3. ✅ Quick test completes without errors
4. ✅ Training shows accuracy improving
5. ✅ Models saved to Drive successfully

---

## 📞 Getting Help

If you encounter issues:

1. **Check this guide** - Read the sections below for detailed instructions

2. **Check the notebook** - `EEG_to_Text_Colab_Updated.ipynb` has troubleshooting cells

3. **Verify checklist** - Make sure all setup items are checked

4. **Check paths** - Most issues are incorrect Drive paths

---

## 🎉 You're Ready!

Everything is set up and ready to go. Choose your path above and start training!

**Estimated total time**: 35-65 minutes (setup + training)

**Expected result**: 55-65% accuracy (172-241x better than random!)

Good luck! 🚀

---

## 📈 What's Next?

After successful training:

1. **Experiment** - Try different hyperparameters
2. **Optimize** - Use optimized settings for better accuracy
3. **Deploy** - Use trained models for predictions
4. **Iterate** - Improve based on results

---

## 📝 Notes

- All code is automatically fetched from GitHub
- Dataset stays in your Google Drive
- Models are saved back to Drive
- Free GPU (T4) is sufficient
- Can resume if disconnected
- No local setup required!

---

**Last Updated**: 2025-11-25

**Repository**: https://github.com/Tejas-Chakkarwar/ML-Project-Data.git

**Dataset**: https://drive.google.com/drive/folders/1R3RAoh-G6Wa3jwpN3_qkYPGcf87K8Lbw
