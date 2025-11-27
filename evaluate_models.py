#!/usr/bin/env python3
"""
Evaluation-Only Script for Trained EEG-to-Text HMM Models
Loads pre-trained CNN and HMM models and evaluates on test set
"""

import sys
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from feature_extractor import CNNEEGEncoder
from config import CNN_INPUT_CHANNELS, SEQUENCE_LENGTH, CNN_HIDDEN_CHANNELS, CNN_DEVICE
import hmm_model

def load_models():
    """Load saved CNN and HMM models"""
    print("=" * 70)
    print("LOADING TRAINED MODELS")
    print("=" * 70)
    
    # Load CNN
    cnn_path = Path('checkpoints/cnn_encoder.pth')
    if not cnn_path.exists():
        raise FileNotFoundError(f"CNN model not found at {cnn_path}")
    
    encoder = CNNEEGEncoder(
        input_channels=CNN_INPUT_CHANNELS,
        sequence_length=SEQUENCE_LENGTH,
        hidden_channels=CNN_HIDDEN_CHANNELS
    ).to(CNN_DEVICE)
    
    checkpoint = torch.load(cnn_path, map_location=CNN_DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        encoder.load_state_dict(checkpoint)
    
    encoder.eval()
    print(f"✓ Loaded CNN from {cnn_path}")
    
    # Load HMMs
    hmm_path = Path('checkpoints/hmm_models.pkl')
    if not hmm_path.exists():
        raise FileNotFoundError(f"HMM models not found at {hmm_path}")
    
    with open(hmm_path, 'rb') as f:
        hmm_data = pickle.load(f)
    
    if isinstance(hmm_data, dict) and 'models' in hmm_data:
        hmm_models = hmm_data['models']
        print(f"✓ Loaded {len(hmm_models)} HMM models from {hmm_path}")
        print(f"  (n_states={hmm_data.get('n_states', 'unknown')}, n_features={hmm_data.get('n_features', 'unknown')})")
    else:
        hmm_models = hmm_data
        print(f"✓ Loaded {len(hmm_models)} HMM models from {hmm_path}")
    
    return encoder, hmm_models

def load_test_data():
    """Load all test data"""
    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)
    
    # Load mapping
    mapping_df = pd.read_csv('processed_data/sentence_mapping.csv')
    
    # Get test split (same logic as training)
    sentence_groups = mapping_df.groupby('Content')
    valid_sentences = [s for s, g in sentence_groups if len(g) >= 3]
    
    print(f"Total sentences with >= 3 samples: {len(valid_sentences)}")
    
    test_data = []
    test_labels = []
    test_files = []
    
    for sentence in valid_sentences:
        group = sentence_groups.get_group(sentence)
        n_samples = len(group)
        n_test = max(1, int(n_samples * 0.2))  # 20% for test
        
        # Take last 20% as test
        test_group = group.iloc[-n_test:]
        
        for _, row in test_group.iterrows():
            filename = row['CSVFilename']
            filepath = Path('processed_data') / filename
            
            if filepath.exists():
                try:
                    data = pd.read_csv(filepath).values
                    test_data.append(data)
                    test_labels.append(sentence)
                    test_files.append(filename)
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
    
    print(f"✓ Loaded {len(test_data)} test samples")
    return test_data, test_labels, test_files

def extract_features(encoder, raw_data_list, batch_size=32):
    """Extract CNN features from raw EEG data"""
    print("\n" + "=" * 70)
    print("EXTRACTING FEATURES")
    print("=" * 70)
    
    features_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(raw_data_list), batch_size), desc="Extracting"):
            batch_data = raw_data_list[i:i+batch_size]
            
            for raw_data in batch_data:
                # Prepare input
                X = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0).to(CNN_DEVICE)
                
                # Extract features
                encoded, _ = encoder(X)
                features = encoded.cpu().numpy().squeeze()
                
                # Transpose to match HMM expectations: (T, features)
                features = features.T
                
                features_list.append(features)
    
    print(f"✓ Extracted features for {len(features_list)} samples")
    return features_list

def predict(hmm_models, features_list):
    """Predict sentences using HMM models"""
    print("\n" + "=" * 70)
    print("MAKING PREDICTIONS")
    print("=" * 70)
    
    predictions = []
    
    for features in tqdm(features_list, desc="Predicting"):
        best_sentence = None
        best_score = float('-inf')
        
        for sentence, hmm in hmm_models.items():
            try:
                score = hmm.score(features)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
            except Exception as e:
                # Skip models that fail to score
                continue
        
        predictions.append(best_sentence)
    
    return predictions

def evaluate(true_labels, predictions, test_files):
    """Evaluate predictions and generate report"""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    # Overall accuracy
    correct = sum(1 for t, p in zip(true_labels, predictions) if t == p)
    total = len(true_labels)
    accuracy = (correct / total) * 100
    
    print(f"\nOverall Accuracy: {correct}/{total} = {accuracy:.2f}%")
    
    # Per-sentence accuracy
    sentence_stats = {}
    for true, pred, file in zip(true_labels, predictions, test_files):
        if true not in sentence_stats:
            sentence_stats[true] = {'correct': 0, 'total': 0}
        
        sentence_stats[true]['total'] += 1
        if true == pred:
            sentence_stats[true]['correct'] += 1
    
    # Show top 10 best and worst performing sentences
    sorted_sentences = sorted(
        sentence_stats.items(),
        key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0,
        reverse=True
    )
    
    print("\n" + "=" * 70)
    print("TOP 10 BEST PERFORMING SENTENCES")
    print("=" * 70)
    for i, (sentence, stats) in enumerate(sorted_sentences[:10], 1):
        acc = (stats['correct'] / stats['total']) * 100
        print(f"{i:2d}. {acc:5.1f}% ({stats['correct']}/{stats['total']}) - {sentence[:50]}...")
    
    print("\n" + "=" * 70)
    print("TOP 10 WORST PERFORMING SENTENCES")
    print("=" * 70)
    for i, (sentence, stats) in enumerate(sorted_sentences[-10:], 1):
        acc = (stats['correct'] / stats['total']) * 100
        print(f"{i:2d}. {acc:5.1f}% ({stats['correct']}/{stats['total']}) - {sentence[:50]}...")
    
    # Show some example predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS (First 10)")
    print("=" * 70)
    for i in range(min(10, len(true_labels))):
        match = "✓" if true_labels[i] == predictions[i] else "✗"
        print(f"\n{match} Sample {i+1}:")
        print(f"  File: {test_files[i]}")
        print(f"  True: {true_labels[i][:60]}...")
        print(f"  Pred: {predictions[i][:60] if predictions[i] else 'None'}...")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'file': test_files,
        'true_sentence': true_labels,
        'predicted_sentence': predictions,
        'correct': [t == p for t, p in zip(true_labels, predictions)]
    })
    
    results_path = 'evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Detailed results saved to {results_path}")
    
    return accuracy

def main():
    print("\n" + "=" * 70)
    print("EEG-TO-TEXT HMM EVALUATION")
    print("=" * 70)
    print("\nThis script evaluates pre-trained models on the test set.")
    print("No training will be performed.\n")
    
    # Load models
    encoder, hmm_models = load_models()
    
    # Load test data
    test_data, test_labels, test_files = load_test_data()
    
    # Extract features
    features_list = extract_features(encoder, test_data)
    
    # Make predictions
    predictions = predict(hmm_models, features_list)
    
    # Evaluate
    accuracy = evaluate(test_labels, predictions, test_files)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Final Accuracy: {accuracy:.2f}%")
    print("=" * 70)

if __name__ == '__main__':
    main()
