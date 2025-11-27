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
from feature_extractor import CNNEEGEncoder, train_autoencoder
from predictor import SentencePredictor
import config
from utils import print_evaluation_summary, format_time

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EEG-to-Text HMM Pipeline (Memory Efficient)')
    
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
    print("EEG-TO-TEXT HMM PIPELINE (MEMORY EFFICIENT)")
    print("=" * 70)
    print()
    
    # Data Directory
    data_dir = os.path.join(os.path.dirname(__file__), config.DATA_DIR)
    
    # ========================================================================
    # STEP 1: LOAD DATA METADATA (NOT ACTUAL DATA YET)
    # ========================================================================
    print("STEP 1: Loading Data Metadata")
    print("-" * 70)
    
    loader = DataLoader(data_dir)
    loader.load_mapping()
    
    # Get file list
    files = loader.get_all_files()
    
    # Quick test mode or max files limit
    if args.quick_test:
        print(f"⚡ Quick test mode: using {config.QUICK_TEST_MAX_FILES} files")
        files = files[:config.QUICK_TEST_MAX_FILES]
        args.cnn_epochs = config.QUICK_TEST_CNN_EPOCHS
        if args.min_samples == config.MIN_SAMPLES_PER_SENTENCE:
            args.min_samples = 2
            print(f"⚡ Adjusted min_samples to 2 for quick test mode")
    elif args.max_files:
        print(f"⚡ Limiting to {args.max_files} files")
        files = files[:args.max_files]
    
    print(f"✓ Will process {len(files)} files")
    print()

    # ========================================================================
    # STEP 2: BUILD SENTENCE INDEX (WITHOUT LOADING DATA)
    # ========================================================================
    print("STEP 2: Building Sentence Index")
    print("-" * 70)
    
    # Map files to sentences
    file_to_text = {}
    text_to_files = {}
    
    for f in files:
        text = loader.get_text_for_file(f)
        if text:
            file_to_text[f] = text
            if text not in text_to_files:
                text_to_files[text] = []
            text_to_files[text].append(f)
    
    # Filter sentences with enough samples
    filtered_text_to_files = {k: v for k, v in text_to_files.items() if len(v) >= args.min_samples}
    
    print(f"✓ Found {len(filtered_text_to_files)} sentences with >= {args.min_samples} samples")
    print(f"  (Total unique sentences: {len(text_to_files)})")
    print()
    
    if len(filtered_text_to_files) == 0:
        print("✗ No sentences found with enough samples. Try lowering --min-samples or loading more files.")
        return

    # ========================================================================
    # STEP 3: CREATE TRAIN/TEST SPLIT (FILE LISTS ONLY)
    # ========================================================================
    print("STEP 3: Creating Train/Test Split")
    print("-" * 70)
    
    train_files_list = []
    test_files_list = []
    
    for text, file_list in filtered_text_to_files.items():
        n = len(file_list)
        n_train = int(n * args.train_split)
        if n_train == 0:
            n_train = 1
        
        train_files_list.extend(file_list[:n_train])
        test_files_list.extend(file_list[n_train:])
    
    print(f"✓ Training files: {len(train_files_list)}")
    print(f"✓ Test files: {len(test_files_list)}")
    print()

    # ========================================================================
    # STEP 4: LOAD AND AUGMENT TRAINING DATA IN BATCHES
    # ========================================================================
    print("STEP 4: Loading and Augmenting Training Data (Batch Processing)")
    print("-" * 70)
    
    # Process in smaller batches to avoid memory issues
    # Batch size reduced from 500 to 50 to prevent memory errors
    batch_size = 50
    augmented_raw_list = []
    augmented_text_list = []
    target_shape = (config.CNN_INPUT_CHANNELS, config.SEQUENCE_LENGTH)
    
    for batch_start in range(0, len(train_files_list), batch_size):
        batch_end = min(batch_start + batch_size, len(train_files_list))
        batch_files = train_files_list[batch_start:batch_end]
        
        print(f"  Processing batch {batch_start//batch_size + 1}/{(len(train_files_list)-1)//batch_size + 1} ({len(batch_files)} files)...")
        
        for f in batch_files:
            data = loader.load_padded_data(f, target_length=config.SEQUENCE_LENGTH)
            text = file_to_text[f]
            
            if data is not None and data.shape == target_shape:
                # Generate augmented versions
                augmented_batch = loader.augment_data(data, num_augmentations=args.num_aug)
                
                for aug in augmented_batch:
                    if aug.shape == target_shape:
                        augmented_raw_list.append(aug)
                        augmented_text_list.append(text)
    
    print(f"✓ Total training samples after augmentation: {len(augmented_raw_list)}")
    print(f"  (Augmentation factor: {len(augmented_raw_list) / len(train_files_list):.1f}x)")
    print()

    # ========================================================================
    # STEP 5: TRAIN CNN AUTOENCODER
    # ========================================================================
    print("STEP 5: Training CNN Autoencoder")
    print("-" * 70)
    
    X_tensor = torch.tensor(np.array(augmented_raw_list), dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    train_loader = TorchDataLoader(dataset, batch_size=args.cnn_batch_size, shuffle=True)
    
    encoder = CNNEEGEncoder(
        input_channels=config.CNN_INPUT_CHANNELS,
        hidden_channels=config.CNN_HIDDEN_CHANNELS,
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
    
    encoder, history = train_autoencoder(
        encoder, 
        train_loader, 
        num_epochs=args.cnn_epochs,
        learning_rate=args.cnn_lr,
        device=config.CNN_DEVICE,
        checkpoint_path=config.CNN_CHECKPOINT_FILE if args.save_models else None
    )
    print()

    # ========================================================================
    # STEP 6: EXTRACT FEATURES FOR HMM
    # ========================================================================
    print("STEP 6: Extracting Features for HMM")
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
    
    # Free memory
    del X_tensor, dataset, train_loader, augmented_raw_list
    
    # Extract features for test data (load in batches)
    print("Extracting test features...")
    test_raw_list = []
    test_text_list = []
    
    for f in test_files_list:
        data = loader.load_padded_data(f, target_length=config.SEQUENCE_LENGTH)
        text = file_to_text[f]
        
        if data is not None and data.shape == target_shape:
            test_raw_list.append(data)
            test_text_list.append(text)
    
    X_test_tensor = torch.tensor(np.array(test_raw_list), dtype=torch.float32)
    with torch.no_grad():
        test_features_tensor = encoder.get_features(X_test_tensor)
    test_features_np = test_features_tensor.numpy()
    hmm_test_list = []
    for j in range(test_features_np.shape[0]):
        hmm_test_list.append(test_features_np[j].T)
    
    print(f"✓ Extracted {len(hmm_test_list)} test feature sequences")
    
    # Free memory
    del X_test_tensor, test_raw_list
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
    print(f"Unique Sentences: {len(filtered_text_to_files)}")
    print(f"Test Samples: {len(hmm_test_list)}")
    
    if args.save_models:
        print()
        print("Saved Models:")
        print(f"  - CNN Encoder: {config.CNN_CHECKPOINT_FILE}")
        print(f"  - HMM Models: {config.HMM_MODELS_FILE}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
