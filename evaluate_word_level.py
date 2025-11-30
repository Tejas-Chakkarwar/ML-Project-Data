"""
Evaluate trained models with word-level metrics

This script loads existing trained models and evaluates them with comprehensive
word-level metrics including:
- Word-level accuracy (% of words correct)
- Word Error Rate (WER)
- Per-sample word accuracy
- Substitution/Deletion/Insertion statistics

Usage:
    python evaluate_word_level.py --cnn-checkpoint checkpoints/cnn_encoder.pth \
                                   --hmm-models checkpoints/hmm_models.pkl \
                                   --data-dir processed_data \
                                   --min-samples 25
"""

import sys
import os
import argparse
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from feature_extractor import SupervisedCNNEncoder
from predictor import SentencePredictor
import config
from utils import print_evaluation_summary, calculate_word_level_metrics
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate models with word-level metrics')

    parser.add_argument('--cnn-checkpoint', type=str, default=config.CNN_CHECKPOINT_FILE,
                       help='Path to CNN checkpoint')
    parser.add_argument('--hmm-models', type=str, default=config.HMM_MODELS_FILE,
                       help='Path to HMM models file')
    parser.add_argument('--data-dir', type=str, default=config.DATA_DIR,
                       help='Path to data directory')
    parser.add_argument('--min-samples', type=int, default=25,
                       help='Minimum samples per sentence to include')
    parser.add_argument('--train-split', type=float, default=config.TRAIN_TEST_SPLIT,
                       help='Train/test split ratio')
    parser.add_argument('--num-aug', type=int, default=1,
                       help='Number of augmentations (for feature extraction)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("WORD-LEVEL EVALUATION")
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

    # STEP 1: Load data metadata
    print("STEP 1: Loading Data Metadata")
    print("-" * 70)

    loader = DataLoader(data_dir)
    loader.load_mapping()
    files = loader.get_all_files()

    print(f"✓ Found {len(files)} total files")
    print()

    # STEP 2: Build sentence index
    print("STEP 2: Building Sentence Index")
    print("-" * 70)

    file_to_text = {}
    text_to_files = {}

    for f in files:
        text = loader.get_text_for_file(f)
        if text:
            file_to_text[f] = text
            if text not in text_to_files:
                text_to_files[text] = []
            text_to_files[text].append(f)

    filtered_text_to_files = {k: v for k, v in text_to_files.items() if len(v) >= args.min_samples}

    print(f"✓ Found {len(filtered_text_to_files)} sentences with >= {args.min_samples} samples")
    print()

    # STEP 3: Train/test split
    print("STEP 3: Creating Train/Test Split")
    print("-" * 70)

    test_files_list = []
    for text, file_list in filtered_text_to_files.items():
        n = len(file_list)
        n_train = max(1, int(n * args.train_split))
        test_files_list.extend(file_list[n_train:])

    print(f"✓ Test files: {len(test_files_list)}")
    print()

    # STEP 4: Load CNN
    print("STEP 4: Loading CNN Encoder")
    print("-" * 70)

    num_classes = len(filtered_text_to_files)
    encoder = SupervisedCNNEncoder(
        input_channels=config.CNN_INPUT_CHANNELS,
        hidden_channels=config.CNN_HIDDEN_CHANNELS,
        num_classes=num_classes,
        sequence_length=config.SEQUENCE_LENGTH
    )

    checkpoint = torch.load(args.cnn_checkpoint, map_location=config.CNN_DEVICE)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.to(config.CNN_DEVICE)
    encoder.eval()

    print(f"✓ Loaded CNN from {args.cnn_checkpoint}")
    print()

    # STEP 5: Extract test features
    print("STEP 5: Extracting Test Features")
    print("-" * 70)

    target_shape = (config.CNN_INPUT_CHANNELS, config.SEQUENCE_LENGTH)
    hmm_test_list = []
    test_text_list = []

    for i, f in enumerate(test_files_list):
        data = loader.load_padded_data(f, target_length=config.SEQUENCE_LENGTH)
        text = file_to_text[f]

        if data is not None and data.shape == target_shape:
            X_test = torch.tensor(np.array([data]), dtype=torch.float32).to(config.CNN_DEVICE)

            with torch.no_grad():
                features = encoder.get_features(X_test)
                features_np = features.cpu().numpy()
                hmm_test_list.append(features_np[0].T)
                test_text_list.append(text)

            del X_test, features, features_np

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(test_files_list)} files")

    print(f"✓ Extracted {len(hmm_test_list)} test sequences")

    # Normalize features (use simple standardization as approximation)
    print("\nNormalizing features...")
    all_test_features = np.vstack([f for f in hmm_test_list])
    scaler = StandardScaler()
    scaler.fit(all_test_features)
    hmm_test_list = [scaler.transform(f) for f in hmm_test_list]
    print("✓ Features normalized")
    print()

    # STEP 6: Load HMM and evaluate
    print("STEP 6: Loading HMM and Evaluating")
    print("-" * 70)

    predictor = SentencePredictor(
        n_states=config.HMM_N_STATES,
        n_features=config.HMM_N_FEATURES
    )
    predictor.load(args.hmm_models)

    print(f"✓ Loaded HMM models from {args.hmm_models}")
    print()

    # Make predictions
    print("Making predictions...")
    true_labels = []
    pred_labels = []

    for i, X in enumerate(hmm_test_list):
        true_text = test_text_list[i]
        pred_text, score = predictor.predict(X)

        true_labels.append(true_text)
        pred_labels.append(pred_text)

        if (i + 1) % 500 == 0:
            print(f"  Predicted {i+1}/{len(hmm_test_list)} samples")

    print()

    # STEP 7: Print evaluation with word-level metrics
    print("STEP 7: Evaluation Results")
    print("-" * 70)

    overall_acc, sentence_acc = print_evaluation_summary(true_labels, pred_labels, verbose=True)

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Test Samples: {len(hmm_test_list)}")
    print(f"Sentence Accuracy: {overall_acc*100:.2f}%")
    print()
    print("💡 TIP: Word-level accuracy shows partial correctness and is more informative")
    print("    than sentence-level accuracy for EEG-to-text tasks!")
    print("=" * 70)


if __name__ == "__main__":
    main()
