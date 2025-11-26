"""
Utility functions for evaluation, visualization, and model management
"""

import os
import pickle
import numpy as np
from collections import defaultdict
import torch


def save_checkpoint(model, filepath, metadata=None):
    """
    Save model checkpoint with optional metadata
    
    Args:
        model: PyTorch model or any picklable object
        filepath: Path to save checkpoint
        metadata: Optional dictionary with additional info
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model=None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: Optional model to load state dict into
        
    Returns:
        Loaded checkpoint or model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {filepath}")
        return model, checkpoint.get('metadata', {})
    
    return checkpoint


def save_hmm_models(models_dict, filepath):
    """
    Save dictionary of HMM models using pickle
    
    Args:
        models_dict: Dictionary mapping text to HMM models
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(models_dict, f)
    
    print(f"✓ Saved {len(models_dict)} HMM models to {filepath}")


def load_hmm_models(filepath):
    """
    Load dictionary of HMM models from pickle file
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Dictionary mapping text to HMM models
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HMM models file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        models_dict = pickle.load(f)
    
    print(f"✓ Loaded {len(models_dict)} HMM models from {filepath}")
    return models_dict


def calculate_confusion_matrix(true_labels, pred_labels, label_to_idx=None):
    """
    Calculate confusion matrix for predictions
    
    Args:
        true_labels: List of true labels (strings)
        pred_labels: List of predicted labels (strings)
        label_to_idx: Optional mapping from label to index
        
    Returns:
        confusion_matrix: numpy array of shape (n_classes, n_classes)
        idx_to_label: List mapping index to label
    """
    # Get unique labels
    unique_labels = sorted(set(true_labels) | set(pred_labels))
    
    if label_to_idx is None:
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    n_classes = len(unique_labels)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for true, pred in zip(true_labels, pred_labels):
        true_idx = label_to_idx.get(true, -1)
        pred_idx = label_to_idx.get(pred, -1)
        
        if true_idx >= 0 and pred_idx >= 0:
            confusion_matrix[true_idx, pred_idx] += 1
    
    return confusion_matrix, idx_to_label


def calculate_per_sentence_accuracy(true_labels, pred_labels):
    """
    Calculate accuracy for each unique sentence
    
    Args:
        true_labels: List of true labels
        pred_labels: List of predicted labels
        
    Returns:
        Dictionary mapping sentence to accuracy
    """
    sentence_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for true, pred in zip(true_labels, pred_labels):
        sentence_stats[true]['total'] += 1
        if true == pred:
            sentence_stats[true]['correct'] += 1
    
    # Calculate accuracy for each sentence
    sentence_accuracy = {}
    for sentence, stats in sentence_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        sentence_accuracy[sentence] = {
            'accuracy': accuracy,
            'correct': stats['correct'],
            'total': stats['total']
        }
    
    return sentence_accuracy


def print_evaluation_summary(true_labels, pred_labels, verbose=True):
    """
    Print comprehensive evaluation summary
    
    Args:
        true_labels: List of true labels
        pred_labels: List of predicted labels
        verbose: If True, print detailed per-sentence accuracy
    """
    # Overall accuracy
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    total = len(true_labels)
    overall_accuracy = correct / total if total > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Overall Accuracy: {correct}/{total} ({overall_accuracy*100:.2f}%)")
    print()
    
    # Per-sentence accuracy
    sentence_accuracy = calculate_per_sentence_accuracy(true_labels, pred_labels)
    
    print(f"Per-Sentence Accuracy ({len(sentence_accuracy)} unique sentences):")
    print("-" * 70)
    
    # Sort by accuracy (lowest first to identify problem sentences)
    sorted_sentences = sorted(sentence_accuracy.items(), 
                             key=lambda x: x[1]['accuracy'])
    
    if verbose:
        for sentence, stats in sorted_sentences:
            acc = stats['accuracy'] * 100
            correct_count = stats['correct']
            total_count = stats['total']
            
            # Truncate long sentences for display
            display_sentence = sentence[:50] + "..." if len(sentence) > 50 else sentence
            
            print(f"  {acc:5.1f}% ({correct_count}/{total_count}) - {display_sentence}")
    else:
        # Just show summary statistics
        accuracies = [s['accuracy'] for s in sentence_accuracy.values()]
        print(f"  Mean: {np.mean(accuracies)*100:.2f}%")
        print(f"  Median: {np.median(accuracies)*100:.2f}%")
        print(f"  Min: {np.min(accuracies)*100:.2f}%")
        print(f"  Max: {np.max(accuracies)*100:.2f}%")
    
    print("=" * 70)
    print()
    
    return overall_accuracy, sentence_accuracy


def format_time(seconds):
    """Format seconds into human-readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"
