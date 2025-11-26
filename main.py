import sys
import os
import numpy as np
import torch
import argparse
import time
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from feature_extractor import CNNEEGEncoder, SupervisedCNNEncoder, train_autoencoder, train_supervised_encoder
from predictor import SentencePredictor
import config
from utils import print_evaluation_summary, format_time

# For feature normalization
from sklearn.preprocessing import StandardScaler

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EEG-to-Text HMM Pipeline')
    
    # Mode
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced dataset')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint file')
    
    # Data parameters
    parser.add_argument('--min-samples', type=int, default=config.MIN_SAMPLES_PER_SENTENCE,
                       help='Minimum samples per sentence')
    parser.add_argument('--train-split', type=float, default=config.TRAIN_TEST_SPLIT,
                       help='Train/test split ratio')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to load (for memory constraints)')
    
    # CNN parameters
    parser.add_argument('--cnn-epochs', type=int, default=config.CNN_EPOCHS,
                       help='Number of CNN training epochs')
    parser.add_argument('--cnn-batch-size', type=int, default=config.CNN_BATCH_SIZE,
                       help='CNN batch size')
    parser.add_argument('--cnn-lr', type=float, default=config.CNN_LEARNING_RATE,
                       help='CNN learning rate')
    
    # HMM parameters
    parser.add_argument('--hmm-states', type=int, default=config.HMM_N_STATES,
                       help='Number of HMM states')
    parser.add_argument('--hmm-iter', type=int, default=config.HMM_MAX_ITER,
                       help='Maximum HMM iterations')
    
    # Augmentation
    parser.add_argument('--num-aug', type=int, default=1,
                       help='Number of augmentations per sample (default: 1 to save memory)')
    
    # Output
    parser.add_argument('--save-models', action='store_true', default=True,
                       help='Save trained models')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress')
    
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    
    print("=" * 70)
    print("EEG-TO-TEXT HMM PIPELINE")
    print("=" * 70)
    print()
    
    # Data Directory
    data_dir = os.path.join(os.path.dirname(__file__), config.DATA_DIR)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("STEP 1: Loading Data")
    print("-" * 70)
    
    loader = DataLoader(data_dir)
    loader.load_mapping()
    
    # Load files for training
    files = loader.get_all_files()
    
    # Quick test mode: limit files
    if args.quick_test:
        print(f"⚡ Quick test mode: using {config.QUICK_TEST_MAX_FILES} files")
        files = files[:config.QUICK_TEST_MAX_FILES]
        args.cnn_epochs = config.QUICK_TEST_CNN_EPOCHS
        # Lower min_samples for quick test since we have fewer files
        if args.min_samples == config.MIN_SAMPLES_PER_SENTENCE:
            args.min_samples = 2
            print(f"⚡ Adjusted min_samples to 2 for quick test mode")
    elif args.max_files:
        print(f"⚡ Limiting to {args.max_files} files (memory constraint)")
        files = files[:args.max_files]
    
    train_files = files
    
    raw_data_list = []
    text_list = []
    
    print(f"Loading {len(train_files)} files...")
    
    for idx, f in enumerate(train_files):
        if idx % 500 == 0 and idx > 0:
            print(f"  Loaded {idx}/{len(train_files)} files...")
        
        # Load padded data (105, 5500)
        data = loader.load_padded_data(f, target_length=config.SEQUENCE_LENGTH)
        text = loader.get_text_for_file(f)
        
        if data is not None and text is not None:
            raw_data_list.append(data)
            text_list.append(text)
    
    print(f"✓ Loaded {len(raw_data_list)} sequences")
    print()
    
    if not raw_data_list:
        print("✗ No data loaded. Exiting.")
        return

    # ========================================================================
    # STEP 2: FILTER FOR CROSS-SUBJECT TRAINING
    # ========================================================================
    print("STEP 2: Filtering for Cross-Subject Training")
    print("-" * 70)
    
    # Group by text
    text_to_data = {}
    for i, raw_data in enumerate(raw_data_list):
        text = text_list[i]
        if text not in text_to_data:
            text_to_data[text] = []
        text_to_data[text].append(raw_data)
    
    # Keep only sentences with >= min_samples
    min_samples = args.min_samples
    filtered_text_to_data = {k: v for k, v in text_to_data.items() if len(v) >= min_samples}
    
    print(f"✓ Found {len(filtered_text_to_data)} sentences with >= {min_samples} samples")
    print(f"  (Total unique sentences: {len(text_to_data)})")
    print()
    
    if len(filtered_text_to_data) == 0:
        print("✗ No sentences found with enough samples. Try lowering --min-samples or loading more files.")
        return

    # ========================================================================
    # STEP 3: TRAIN/TEST SPLIT
    # ========================================================================
    print("STEP 3: Creating Train/Test Split")
    print("-" * 70)
    
    train_raw_list = []
    train_text_list = []
    test_raw_list = []
    test_text_list = []
    
    for text, data_list in filtered_text_to_data.items():
        # Split: train_split% Train, rest Test
        n = len(data_list)
        n_train = int(n * args.train_split)
        if n_train == 0: 
            n_train = 1  # Ensure at least 1 train sample
        
        train_data = data_list[:n_train]
        test_data = data_list[n_train:]
        
        train_raw_list.extend(train_data)
        train_text_list.extend([text] * len(train_data))
        
        test_raw_list.extend(test_data)
        test_text_list.extend([text] * len(test_data))
    
    print(f"✓ Training Set: {len(train_raw_list)} samples")
    print(f"✓ Test Set: {len(test_raw_list)} samples")
    print()

    # ========================================================================
    # STEP 4: DATA AUGMENTATION
    # ========================================================================
    print("STEP 4: Augmenting Training Data")
    print("-" * 70)
    
    augmented_raw_list = []
    augmented_text_list = []
    
    target_shape = (config.CNN_INPUT_CHANNELS, config.SEQUENCE_LENGTH)
    
    for i, raw_data in enumerate(train_raw_list):
        if i % 500 == 0 and i > 0:
            print(f"  Augmented {i}/{len(train_raw_list)} samples...")
        
        # Verify shape
        if raw_data.shape != target_shape:
            # Re-pad if necessary
            if raw_data.shape[1] < config.SEQUENCE_LENGTH:
                pad_width = config.SEQUENCE_LENGTH - raw_data.shape[1]
                raw_data = np.pad(raw_data, ((0, 0), (0, pad_width)), mode='constant')
            elif raw_data.shape[1] > config.SEQUENCE_LENGTH:
                raw_data = raw_data[:, :config.SEQUENCE_LENGTH]
        
        text = train_text_list[i]
        # Generate augmented versions
        augmented_batch = loader.augment_data(raw_data, num_augmentations=args.num_aug)
        
        for aug in augmented_batch:
            if aug.shape == target_shape:
                augmented_raw_list.append(aug)
                augmented_text_list.append(text)
    
    print(f"✓ Total training samples after augmentation: {len(augmented_raw_list)}")
    print(f"  (Augmentation factor: {len(augmented_raw_list) / len(train_raw_list):.1f}x)")
    print()

    # ========================================================================
    # STEP 5: TRAIN CNN ENCODER (SUPERVISED)
    # ========================================================================
    print("STEP 5: Training CNN Encoder (Supervised)")
    print("-" * 70)

    # Create label mapping: sentence -> integer label
    unique_sentences = list(filtered_text_to_data.keys())
    sentence_to_label = {sent: idx for idx, sent in enumerate(unique_sentences)}
    num_classes = len(unique_sentences)
    print(f"Number of unique classes: {num_classes}")

    # Create labels for augmented data
    augmented_labels = [sentence_to_label[text] for text in augmented_text_list]

    # Create dataset with labels
    X_tensor = torch.tensor(np.array(augmented_raw_list), dtype=torch.float32)
    y_tensor = torch.tensor(augmented_labels, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = TorchDataLoader(dataset, batch_size=args.cnn_batch_size, shuffle=True)

    # Use supervised CNN encoder
    encoder = SupervisedCNNEncoder(
        input_channels=config.CNN_INPUT_CHANNELS,
        hidden_channels=config.CNN_HIDDEN_CHANNELS,
        num_classes=num_classes,
        sequence_length=config.SEQUENCE_LENGTH
    )

    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=config.CNN_DEVICE)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Checkpoint loaded")

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    encoder, history = train_supervised_encoder(
        encoder,
        train_loader,
        num_epochs=args.cnn_epochs,
        learning_rate=args.cnn_lr,
        device=config.CNN_DEVICE,
        checkpoint_path=config.CNN_CHECKPOINT_FILE if args.save_models else None
    )
    print()

    # ========================================================================
    # STEP 6: EXTRACT FEATURES FOR HMM + NORMALIZATION
    # ========================================================================
    print("STEP 6: Extracting Features for HMM + Normalization")
    print("-" * 70)

    encoder.eval()

    # Extract features for training data
    print("Extracting training features...")
    hmm_train_list = []
    extract_loader = TorchDataLoader(dataset, batch_size=config.EXTRACT_BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(extract_loader):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(extract_loader)}...")

            inputs = batch[0]
            features = encoder.get_features(inputs)
            features_np = features.numpy()
            for j in range(features_np.shape[0]):
                hmm_train_list.append(features_np[j].T)

    print(f"✓ Extracted {len(hmm_train_list)} training feature sequences")

    # Extract features for test data
    print("Extracting test features...")
    X_test_tensor = torch.tensor(np.array(test_raw_list), dtype=torch.float32)
    with torch.no_grad():
        test_features_tensor = encoder.get_features(X_test_tensor)
    test_features_np = test_features_tensor.numpy()
    hmm_test_list = []
    for j in range(test_features_np.shape[0]):
        hmm_test_list.append(test_features_np[j].T)

    print(f"✓ Extracted {len(hmm_test_list)} test feature sequences")

    # ========================================================================
    # FEATURE NORMALIZATION (CRITICAL for HMM stability)
    # ========================================================================
    print("\nNormalizing features...")

    # Concatenate all training features to fit scaler
    all_train_features = np.vstack([f for f in hmm_train_list])

    # Fit StandardScaler on training data
    scaler = StandardScaler()
    scaler.fit(all_train_features)

    # Transform both train and test features
    hmm_train_list = [scaler.transform(f) for f in hmm_train_list]
    hmm_test_list = [scaler.transform(f) for f in hmm_test_list]

    print(f"✓ Features normalized (mean=0, std=1)")
    print()

    # ========================================================================
    # STEP 7: TRAIN HMM SENTENCE PREDICTOR
    # ========================================================================
    print("STEP 7: Training HMM Sentence Predictor")
    print("-" * 70)
    
    predictor = SentencePredictor(
        n_states=args.hmm_states,
        n_features=config.HMM_N_FEATURES
    )
    predictor.train(hmm_train_list, augmented_text_list, verbose=args.verbose)
    
    # Save HMM models
    if args.save_models:
        predictor.save(config.HMM_MODELS_FILE)
    print()

    # ========================================================================
    # STEP 8: EVALUATE ON TEST SET
    # ========================================================================
    print("STEP 8: Evaluating on Test Set")
    print("-" * 70)
    
    if len(hmm_test_list) == 0:
        print("✗ Test set is empty (likely due to small sample size per sentence).")
        return
    
    true_labels = []
    pred_labels = []
    
    print(f"Running predictions on {len(hmm_test_list)} test samples...")
    print()
    
    for i, X in enumerate(hmm_test_list):
        true_text = test_text_list[i]
        pred_text, score = predictor.predict(X)
        
        true_labels.append(true_text)
        pred_labels.append(pred_text)
        
        is_correct = (pred_text == true_text)
        
        if args.verbose and i < 10:  # Show first 10 predictions
            print(f"Sample {i+1}:")
            print(f"  True: {true_text[:60]}...")
            print(f"  Pred: {pred_text[:60] if pred_text else 'None'}... (Score: {score:.2f})")
            print(f"  Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            print()
    
    # Print comprehensive evaluation
    overall_acc, sentence_acc = print_evaluation_summary(true_labels, pred_labels, verbose=True)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    elapsed_time = time.time() - start_time
    
    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)
    print(f"Total Time: {format_time(elapsed_time)}")
    print(f"Final Accuracy: {overall_acc*100:.2f}%")
    print(f"Unique Sentences: {len(filtered_text_to_data)}")
    print(f"Test Samples: {len(hmm_test_list)}")
    
    if args.save_models:
        print()
        print("Saved Models:")
        print(f"  - CNN Encoder: {config.CNN_CHECKPOINT_FILE}")
        print(f"  - HMM Models: {config.HMM_MODELS_FILE}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
