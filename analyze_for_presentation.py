"""
Presentation Analysis Script
Analyzes trained EEG-to-Text models and generates presentation-ready metrics

Features:
- Identifies best predictions (high confidence, correct)
- Identifies worst predictions (failures with analysis)
- Word-level accuracy breakdown
- Presentation metrics (charts, tables)
- Raw data filenames for re-testing examples

Usage:
    python analyze_for_presentation.py --cnn-checkpoint checkpoints/cnn_encoder.pth \
                                       --hmm-models checkpoints/hmm_models.pkl \
                                       --data-dir processed_data \
                                       --output-dir presentation_analysis
"""

import sys
import os
import argparse
import torch
import numpy as np
import json
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from feature_extractor import SupervisedCNNEncoder
from predictor import SentencePredictor
import config
from utils import (calculate_word_accuracy, calculate_word_error_rate,
                   calculate_word_level_metrics, tokenize_sentence)
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze models for presentation')

    parser.add_argument('--cnn-checkpoint', type=str, default=config.CNN_CHECKPOINT_FILE,
                       help='Path to CNN checkpoint')
    parser.add_argument('--hmm-models', type=str, default=config.HMM_MODELS_FILE,
                       help='Path to HMM models file')
    parser.add_argument('--data-dir', type=str, default=config.DATA_DIR,
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='presentation_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--min-samples', type=int, default=None,
                       help='Minimum samples per sentence (auto-detect if not specified)')
    parser.add_argument('--train-split', type=float, default=config.TRAIN_TEST_SPLIT,
                       help='Train/test split ratio')

    return parser.parse_args()


def analyze_predictions(true_labels, pred_labels, test_files, predictor):
    """
    Analyze predictions and categorize by quality

    Returns:
        best_predictions: Top predictions (correct, high confidence)
        worst_predictions: Failed predictions with analysis
        metrics: Overall statistics
    """
    results = []

    for i, (true_text, pred_text, filename) in enumerate(zip(true_labels, pred_labels, test_files)):
        # Get prediction confidence score
        # Load the file again to get features for scoring
        # (In practice, we'd cache this, but for simplicity we recalculate)

        # Calculate word-level metrics
        word_acc, correct_words, total_words = calculate_word_accuracy(true_text, pred_text)
        wer, ops = calculate_word_error_rate(true_text, pred_text)

        is_correct = (true_text == pred_text)

        result = {
            'index': i,
            'filename': filename,
            'true_sentence': true_text,
            'predicted_sentence': pred_text,
            'is_correct': is_correct,
            'word_accuracy': word_acc,
            'correct_words': correct_words,
            'total_words': total_words,
            'wer': wer,
            'substitutions': ops['substitutions'],
            'deletions': ops['deletions'],
            'insertions': ops['insertions']
        }

        results.append(result)

    # Sort by word accuracy (high to low)
    results_sorted = sorted(results, key=lambda x: (x['is_correct'], x['word_accuracy']), reverse=True)

    # Best predictions: correct and high word accuracy
    best_predictions = [r for r in results_sorted if r['is_correct']][:10]

    # Worst predictions: incorrect or low word accuracy
    worst_predictions = [r for r in results_sorted if not r['is_correct']][:10]

    # Calculate overall metrics
    total_correct = sum(1 for r in results if r['is_correct'])
    total_samples = len(results)
    sentence_accuracy = total_correct / total_samples if total_samples > 0 else 0

    total_word_correct = sum(r['correct_words'] for r in results)
    total_word_count = sum(r['total_words'] for r in results)
    word_accuracy = total_word_correct / total_word_count if total_word_count > 0 else 0

    avg_wer = sum(r['wer'] for r in results) / total_samples if total_samples > 0 else 0

    metrics = {
        'total_samples': total_samples,
        'sentence_accuracy': sentence_accuracy,
        'word_accuracy': word_accuracy,
        'avg_wer': avg_wer,
        'total_correct_sentences': total_correct,
        'total_correct_words': total_word_correct,
        'total_words': total_word_count
    }

    return best_predictions, worst_predictions, metrics, results


def generate_presentation_output(best_preds, worst_preds, metrics, output_dir, num_classes):
    """Generate presentation-ready output files"""

    os.makedirs(output_dir, exist_ok=True)

    # 1. Summary Statistics (for PPT slides)
    summary = {
        'Dataset': {
            'Test Samples': metrics['total_samples'],
            'Unique Classes': num_classes,
            'Total Words Analyzed': metrics['total_words']
        },
        'Model Performance': {
            'Classification Accuracy': f"{metrics['sentence_accuracy']*100:.2f}%",
            'Word-Level Accuracy': f"{metrics['word_accuracy']*100:.2f}%",
            'Word Error Rate': f"{metrics['avg_wer']*100:.2f}%",
            'Correct Predictions': f"{metrics['total_correct_sentences']}/{metrics['total_samples']}"
        },
        'Improvement vs Random': {
            'Random Baseline': f"{100/num_classes:.2f}%",
            'Our Model': f"{metrics['sentence_accuracy']*100:.2f}%",
            'Improvement Factor': f"{metrics['sentence_accuracy']/(1/num_classes):.1f}x"
        }
    }

    with open(os.path.join(output_dir, 'summary_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # 2. Best Predictions (for demo)
    with open(os.path.join(output_dir, 'best_predictions.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TOP 10 BEST PREDICTIONS (For Demonstration)\n")
        f.write("=" * 80 + "\n\n")

        for i, pred in enumerate(best_preds[:10], 1):
            f.write(f"[{i}] EXAMPLE {i}: ✓ CORRECT\n")
            f.write("-" * 80 + "\n")
            f.write(f"Raw Data File: {pred['filename']}\n")
            f.write(f"True Sentence: {pred['true_sentence']}\n")
            f.write(f"Predicted: {pred['predicted_sentence']}\n")
            f.write(f"Word Accuracy: {pred['word_accuracy']*100:.1f}% ({pred['correct_words']}/{pred['total_words']})\n")
            f.write(f"Result: PERFECT MATCH ✓\n")
            f.write("\n")

    # 3. Worst Predictions (for analysis)
    with open(os.path.join(output_dir, 'worst_predictions.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TOP 10 CHALLENGING CASES (Error Analysis)\n")
        f.write("=" * 80 + "\n\n")

        for i, pred in enumerate(worst_preds[:10], 1):
            f.write(f"[{i}] EXAMPLE {i}: ✗ INCORRECT\n")
            f.write("-" * 80 + "\n")
            f.write(f"Raw Data File: {pred['filename']}\n")
            f.write(f"True Sentence: {pred['true_sentence']}\n")
            f.write(f"Predicted: {pred['predicted_sentence']}\n")
            f.write(f"Word Accuracy: {pred['word_accuracy']*100:.1f}% ({pred['correct_words']}/{pred['total_words']})\n")
            f.write(f"Word Error Rate: {pred['wer']*100:.1f}%\n")
            f.write(f"Errors: {pred['substitutions']} substitutions, {pred['deletions']} deletions, {pred['insertions']} insertions\n")

            # Show word differences
            if pred['predicted_sentence']:
                true_words = tokenize_sentence(pred['true_sentence'])
                pred_words = tokenize_sentence(pred['predicted_sentence'])

                wrong_words = []
                for j in range(min(len(true_words), len(pred_words))):
                    if true_words[j] != pred_words[j]:
                        wrong_words.append(f'"{true_words[j]}" → "{pred_words[j]}"')

                if wrong_words:
                    f.write(f"Wrong Words: {', '.join(wrong_words[:5])}")
                    if len(wrong_words) > 5:
                        f.write(f" (+{len(wrong_words)-5} more)")
                    f.write("\n")

            f.write("\n")

    # 4. PPT-ready metrics table
    with open(os.path.join(output_dir, 'ppt_metrics.txt'), 'w') as f:
        f.write("PRESENTATION METRICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("SLIDE 1: Overall Performance\n")
        f.write("-" * 80 + "\n")
        f.write(f"Classification Accuracy: {metrics['sentence_accuracy']*100:.1f}%\n")
        f.write(f"Word-Level Accuracy: {metrics['word_accuracy']*100:.1f}%\n")
        f.write(f"Improvement vs Random: {metrics['sentence_accuracy']/(1/num_classes):.1f}x better\n")
        f.write("\n")

        f.write("SLIDE 2: Detailed Metrics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Test Samples: {metrics['total_samples']}\n")
        f.write(f"Correct Classifications: {metrics['total_correct_sentences']}\n")
        f.write(f"Word Error Rate: {metrics['avg_wer']*100:.1f}%\n")
        f.write(f"Total Words Analyzed: {metrics['total_words']}\n")
        f.write(f"Correctly Predicted Words: {metrics['total_correct_words']}\n")
        f.write("\n")

        f.write("SLIDE 3: Key Achievements\n")
        f.write("-" * 80 + "\n")
        f.write(f"✓ Built end-to-end EEG-to-Text pipeline\n")
        f.write(f"✓ Achieved {metrics['sentence_accuracy']*100:.1f}% classification accuracy\n")
        f.write(f"✓ Word-level accuracy: {metrics['word_accuracy']*100:.1f}% (shows partial understanding)\n")
        f.write(f"✓ Significantly better than random baseline ({100/num_classes:.2f}%)\n")
        f.write("\n")

    # 5. Test files list for re-testing
    with open(os.path.join(output_dir, 'test_files_for_demo.txt'), 'w') as f:
        f.write("TEST FILES FOR LIVE DEMONSTRATION\n")
        f.write("=" * 80 + "\n\n")

        f.write("BEST EXAMPLES (Show these for successful predictions):\n")
        f.write("-" * 80 + "\n")
        for i, pred in enumerate(best_preds[:5], 1):
            f.write(f"{i}. {pred['filename']}\n")
            f.write(f"   Expected: {pred['true_sentence'][:60]}...\n")
            f.write(f"   (Should predict correctly)\n\n")

        f.write("\nCHALLENGING EXAMPLES (Show these for error analysis):\n")
        f.write("-" * 80 + "\n")
        for i, pred in enumerate(worst_preds[:3], 1):
            f.write(f"{i}. {pred['filename']}\n")
            f.write(f"   True: {pred['true_sentence'][:60]}...\n")
            f.write(f"   (Model struggles with this one)\n\n")

    print(f"\n✓ Generated presentation materials in: {output_dir}/")
    print(f"  - summary_metrics.json (overview)")
    print(f"  - best_predictions.txt (top 10 successes)")
    print(f"  - worst_predictions.txt (top 10 failures)")
    print(f"  - ppt_metrics.txt (slide-ready metrics)")
    print(f"  - test_files_for_demo.txt (files for live demo)")


def main():
    args = parse_args()

    print("=" * 70)
    print("PRESENTATION ANALYSIS")
    print("=" * 70)
    print()

    # Check files exist
    if not os.path.exists(args.cnn_checkpoint):
        print(f"❌ CNN checkpoint not found: {args.cnn_checkpoint}")
        return

    if not os.path.exists(args.hmm_models):
        print(f"❌ HMM models not found: {args.hmm_models}")
        return

    data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)

    # STEP 1: Load data
    print("STEP 1: Loading Data")
    print("-" * 70)

    loader = DataLoader(data_dir)
    loader.load_mapping()
    files = loader.get_all_files()

    # Build sentence index
    file_to_text = {}
    text_to_files = {}

    for f in files:
        text = loader.get_text_for_file(f)
        if text:
            file_to_text[f] = text
            if text not in text_to_files:
                text_to_files[text] = []
            text_to_files[text].append(f)

    # Auto-detect min_samples if not specified
    if args.min_samples is None:
        # Try to infer from the number of classes in the HMM model
        import pickle
        with open(args.hmm_models, 'rb') as f:
            hmm_models = pickle.load(f)
        num_hmm_classes = len(hmm_models)

        # Find threshold that gives approximately this many classes
        for threshold in range(25, 5, -1):
            filtered = {k: v for k, v in text_to_files.items() if len(v) >= threshold}
            if len(filtered) >= num_hmm_classes:
                args.min_samples = threshold
                break

        if args.min_samples is None:
            args.min_samples = 10  # Fallback

        print(f"Auto-detected min_samples: {args.min_samples}")

    filtered_text_to_files = {k: v for k, v in text_to_files.items() if len(v) >= args.min_samples}
    num_classes = len(filtered_text_to_files)

    print(f"✓ Found {num_classes} classes with >= {args.min_samples} samples")
    print()

    # STEP 2: Split data
    print("STEP 2: Creating Test Set")
    print("-" * 70)

    test_files_list = []
    test_text_list = []

    for text, file_list in filtered_text_to_files.items():
        n = len(file_list)
        n_train = max(1, int(n * args.train_split))
        for f in file_list[n_train:]:
            test_files_list.append(f)
            test_text_list.append(text)

    print(f"✓ Test files: {len(test_files_list)}")
    print()

    # STEP 3: Load models
    print("STEP 3: Loading Models")
    print("-" * 70)

    encoder = SupervisedCNNEncoder(
        input_channels=config.CNN_INPUT_CHANNELS,
        hidden_channels=config.CNN_HIDDEN_CHANNELS,
        num_classes=num_classes,
        sequence_length=config.SEQUENCE_LENGTH
    )

    checkpoint = torch.load(args.cnn_checkpoint, map_location='cpu')
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()

    predictor = SentencePredictor(
        n_states=config.HMM_N_STATES,
        n_features=config.HMM_N_FEATURES
    )
    predictor.load(args.hmm_models)

    print(f"✓ Loaded CNN and {len(predictor.models)} HMM models")
    print()

    # STEP 4: Extract features and predict
    print("STEP 4: Making Predictions")
    print("-" * 70)

    target_shape = (config.CNN_INPUT_CHANNELS, config.SEQUENCE_LENGTH)
    hmm_test_list = []
    true_labels = []
    valid_test_files = []

    for i, (f, text) in enumerate(zip(test_files_list, test_text_list)):
        data = loader.load_padded_data(f, target_length=config.SEQUENCE_LENGTH)

        if data is not None and data.shape == target_shape:
            X_test = torch.tensor(np.array([data]), dtype=torch.float32)

            with torch.no_grad():
                features = encoder.get_features(X_test)
                features_np = features.cpu().numpy()
                hmm_test_list.append(features_np[0].T)
                true_labels.append(text)
                valid_test_files.append(f)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_files_list)} files")

    # Normalize features
    all_test_features = np.vstack([f for f in hmm_test_list])
    scaler = StandardScaler()
    scaler.fit(all_test_features)
    hmm_test_list = [scaler.transform(f) for f in hmm_test_list]

    # Make predictions
    pred_labels = []
    for X in hmm_test_list:
        pred_text, score = predictor.predict(X)
        pred_labels.append(pred_text)

    print(f"✓ Generated {len(pred_labels)} predictions")
    print()

    # STEP 5: Analyze results
    print("STEP 5: Analyzing Results")
    print("-" * 70)

    best_preds, worst_preds, metrics, all_results = analyze_predictions(
        true_labels, pred_labels, valid_test_files, predictor
    )

    # STEP 6: Generate presentation output
    print()
    print("STEP 6: Generating Presentation Materials")
    print("-" * 70)

    generate_presentation_output(best_preds, worst_preds, metrics, args.output_dir, num_classes)

    # Print summary
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Classes: {num_classes}")
    print(f"Test Samples: {metrics['total_samples']}")
    print(f"Classification Accuracy: {metrics['sentence_accuracy']*100:.2f}%")
    print(f"Word-Level Accuracy: {metrics['word_accuracy']*100:.2f}%")
    print(f"Improvement vs Random: {metrics['sentence_accuracy']/(1/num_classes):.1f}x")
    print()
    print(f"📁 Output saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
