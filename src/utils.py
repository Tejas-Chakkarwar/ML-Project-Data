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
    Print comprehensive evaluation summary with both sentence-level and word-level metrics

    Args:
        true_labels: List of true labels
        pred_labels: List of predicted labels
        verbose: If True, print detailed per-sentence accuracy and examples
    """
    # Sentence-level accuracy
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    total = len(true_labels)
    overall_accuracy = correct / total if total > 0 else 0.0

    # Word-level metrics
    word_metrics = calculate_word_level_metrics(true_labels, pred_labels)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Overall metrics
    print("\n📊 OVERALL METRICS:")
    print("-" * 70)
    print(f"  Sentence Accuracy:  {correct}/{total} ({overall_accuracy*100:.2f}%)")
    print(f"  Word Accuracy:      {word_metrics['total_correct_words']}/{word_metrics['total_words']} ({word_metrics['word_accuracy']*100:.2f}%) ⭐")
    print(f"  Word Error Rate:    {word_metrics['avg_wer']*100:.2f}%")
    print()
    print("  Error Breakdown:")
    print(f"    Substitutions: {word_metrics['total_substitutions']}")
    print(f"    Deletions:     {word_metrics['total_deletions']}")
    print(f"    Insertions:    {word_metrics['total_insertions']}")
    print()

    # Show example predictions
    if verbose:
        print("📝 EXAMPLE PREDICTIONS (First 10):")
        print("-" * 70)

        for i, result in enumerate(word_metrics['per_sample_results'][:10]):
            true_sent = result['true']
            pred_sent = result['pred'] if result['pred'] else "[NO PREDICTION]"
            word_acc = result['word_accuracy'] * 100
            correct_w = result['correct_words']
            total_w = result['total_words']

            # Truncate long sentences
            true_display = true_sent[:60] + "..." if len(true_sent) > 60 else true_sent
            pred_display = pred_sent[:60] + "..." if len(pred_sent) > 60 else pred_sent

            sentence_match = "✓ CORRECT" if true_sent == pred_sent else "✗ WRONG"

            print(f"\n[{i+1}] True: {true_display}")
            print(f"    Pred: {pred_display}")
            print(f"    Word Acc: {word_acc:.1f}% ({correct_w}/{total_w}) | {sentence_match}")

            # Show which words were wrong
            if true_sent != pred_sent and result['pred']:
                true_words = tokenize_sentence(true_sent)
                pred_words = tokenize_sentence(pred_sent)

                wrong_words = []
                for j in range(min(len(true_words), len(pred_words))):
                    if true_words[j] != pred_words[j]:
                        wrong_words.append(f'"{true_words[j]}" → "{pred_words[j]}"')

                if wrong_words:
                    print(f"    Wrong: {', '.join(wrong_words[:3])}", end="")
                    if len(wrong_words) > 3:
                        print(f" (+{len(wrong_words)-3} more)", end="")
                    print()

        print()

    # Per-sentence accuracy distribution
    sentence_accuracy = calculate_per_sentence_accuracy(true_labels, pred_labels)

    print("📈 PER-SENTENCE STATISTICS:")
    print("-" * 70)
    print(f"  Unique sentences: {len(sentence_accuracy)}")

    accuracies = [s['accuracy'] for s in sentence_accuracy.values()]
    print(f"  Mean accuracy:    {np.mean(accuracies)*100:.2f}%")
    print(f"  Median accuracy:  {np.median(accuracies)*100:.2f}%")
    print(f"  Min accuracy:     {np.min(accuracies)*100:.2f}%")
    print(f"  Max accuracy:     {np.max(accuracies)*100:.2f}%")

    print("\n" + "=" * 70)
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


def tokenize_sentence(sentence):
    """
    Simple word tokenizer that splits on whitespace and removes punctuation

    Args:
        sentence: String to tokenize

    Returns:
        List of words (lowercase, no punctuation)
    """
    import string
    # Remove punctuation and convert to lowercase
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # Split on whitespace
    words = sentence.lower().split()
    return words


def calculate_word_accuracy(true_sentence, pred_sentence):
    """
    Calculate word-level accuracy between two sentences

    Args:
        true_sentence: Ground truth sentence (string)
        pred_sentence: Predicted sentence (string)

    Returns:
        word_accuracy: Fraction of words that match (0.0 to 1.0)
        correct_words: Number of correct words
        total_words: Total number of words in true sentence
    """
    if pred_sentence is None or pred_sentence == "":
        return 0.0, 0, len(tokenize_sentence(true_sentence))

    true_words = tokenize_sentence(true_sentence)
    pred_words = tokenize_sentence(pred_sentence)

    total_words = len(true_words)
    if total_words == 0:
        return 1.0, 0, 0

    # Count matching words at each position
    correct_words = 0
    for i in range(min(len(true_words), len(pred_words))):
        if true_words[i] == pred_words[i]:
            correct_words += 1

    word_accuracy = correct_words / total_words
    return word_accuracy, correct_words, total_words


def calculate_word_error_rate(true_sentence, pred_sentence):
    """
    Calculate Word Error Rate (WER) using Levenshtein distance

    WER = (S + D + I) / N
    where:
        S = substitutions
        D = deletions
        I = insertions
        N = number of words in reference

    Args:
        true_sentence: Ground truth sentence (string)
        pred_sentence: Predicted sentence (string)

    Returns:
        wer: Word Error Rate (can be > 1.0 if many insertions)
        operations: Dict with counts of substitutions, deletions, insertions
    """
    if pred_sentence is None or pred_sentence == "":
        true_words = tokenize_sentence(true_sentence)
        return 1.0, {'substitutions': 0, 'deletions': len(true_words), 'insertions': 0}

    true_words = tokenize_sentence(true_sentence)
    pred_words = tokenize_sentence(pred_sentence)

    # Compute Levenshtein distance
    n, m = len(true_words), len(pred_words)

    # Create DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize base cases
    for i in range(n + 1):
        dp[i][0] = i  # All deletions
    for j in range(m + 1):
        dp[0][j] = j  # All insertions

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if true_words[i-1] == pred_words[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Deletion
                    dp[i][j-1],      # Insertion
                    dp[i-1][j-1]     # Substitution
                )

    edit_distance = dp[n][m]

    # Backtrack to count operation types
    i, j = n, m
    substitutions = deletions = insertions = 0

    while i > 0 or j > 0:
        if i == 0:
            insertions += 1
            j -= 1
        elif j == 0:
            deletions += 1
            i -= 1
        elif true_words[i-1] == pred_words[j-1]:
            i -= 1
            j -= 1
        else:
            if dp[i][j] == dp[i-1][j-1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + 1:
                deletions += 1
                i -= 1
            else:
                insertions += 1
                j -= 1

    wer = edit_distance / n if n > 0 else 0.0
    operations = {
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions
    }

    return wer, operations


def calculate_word_level_metrics(true_labels, pred_labels):
    """
    Calculate comprehensive word-level metrics across all predictions

    Args:
        true_labels: List of true sentences
        pred_labels: List of predicted sentences

    Returns:
        metrics: Dictionary with word-level statistics
    """
    total_words = 0
    correct_words = 0
    total_wer = 0.0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0

    per_sample_results = []

    for true_sent, pred_sent in zip(true_labels, pred_labels):
        # Word accuracy
        word_acc, correct, total = calculate_word_accuracy(true_sent, pred_sent)
        total_words += total
        correct_words += correct

        # WER
        wer, ops = calculate_word_error_rate(true_sent, pred_sent)
        total_wer += wer * total  # Weight by number of words
        total_substitutions += ops['substitutions']
        total_deletions += ops['deletions']
        total_insertions += ops['insertions']

        per_sample_results.append({
            'true': true_sent,
            'pred': pred_sent,
            'word_accuracy': word_acc,
            'correct_words': correct,
            'total_words': total,
            'wer': wer,
            'operations': ops
        })

    # Aggregate metrics
    avg_word_accuracy = correct_words / total_words if total_words > 0 else 0.0
    avg_wer = total_wer / total_words if total_words > 0 else 0.0

    metrics = {
        'word_accuracy': avg_word_accuracy,
        'total_correct_words': correct_words,
        'total_words': total_words,
        'avg_wer': avg_wer,
        'total_substitutions': total_substitutions,
        'total_deletions': total_deletions,
        'total_insertions': total_insertions,
        'per_sample_results': per_sample_results
    }

    return metrics


# Alias for backward compatibility with Seq2Seq code
def calculate_wer(true_sentence, pred_sentence):
    """
    Alias for calculate_word_error_rate.
    Returns only the WER value (not the operations dict).

    Args:
        true_sentence: Ground truth sentence (string)
        pred_sentence: Predicted sentence (string)

    Returns:
        wer: Word Error Rate value
    """
    wer, _ = calculate_word_error_rate(true_sentence, pred_sentence)
    return wer
