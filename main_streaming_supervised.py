import sys
import os
import numpy as np
import torch
import argparse
import time
import random
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from feature_extractor import SupervisedCNNEncoder  # CHANGED: Use supervised instead of autoencoder
from predictor import SentencePredictor
import config
from utils import print_evaluation_summary, format_time

# ADDED: Import StandardScaler for feature normalization
from sklearn.preprocessing import StandardScaler

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EEG-to-Text HMM Pipeline (Streaming with Supervised CNN)')

    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced dataset')
    parser.add_argument('--min-samples', type=int, default=config.MIN_SAMPLES_PER_SENTENCE,
                       help='Minimum samples per sentence')
    parser.add_argument('--train-split', type=float, default=config.TRAIN_TEST_SPLIT,
                       help='Train/test split ratio')
    parser.add_argument('--cnn-epochs', type=int, default=config.CNN_EPOCHS,
                       help='Number of CNN training epochs')
    parser.add_argument('--cnn-batch-size', type=int, default=config.CNN_BATCH_SIZE,
                       help='CNN batch size')
    parser.add_argument('--cnn-lr', type=float, default=config.CNN_LEARNING_RATE,
                       help='CNN learning rate')
    parser.add_argument('--hmm-states', type=int, default=config.HMM_N_STATES,
                       help='Number of HMM states')
    parser.add_argument('--hmm-iter', type=int, default=config.HMM_MAX_ITER,
                       help='Maximum HMM iterations')
    parser.add_argument('--num-aug', type=int, default=1,
                       help='Number of augmentations per sample')
    parser.add_argument('--chunk-size', type=int, default=200,
                       help='Number of files to process per chunk')
    parser.add_argument('--save-models', action='store_true', default=True,
                       help='Save trained models')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress')

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("=" * 70)
    print("EEG-TO-TEXT HMM PIPELINE (SUPERVISED STREAMING VERSION - v2)")
    print("=" * 70)
    print()
    print("🔧 Fixes applied:")
    print("  ✓ Chunk shuffling between epochs")
    print("  ✓ Adaptive learning rate (ReduceLROnPlateau)")
    print("  ✓ Lower initial LR for stability")
    print()

    data_dir = os.path.join(os.path.dirname(__file__), config.DATA_DIR)

    # STEP 1: Load metadata (only file paths, not data!)
    print("STEP 1: Loading Data Metadata")
    print("-" * 70)

    loader = DataLoader(data_dir)
    loader.load_mapping()
    files = loader.get_all_files()

    if args.quick_test:
        print(f"⚡ Quick test mode: using {config.QUICK_TEST_MAX_FILES} files")
        files = files[:config.QUICK_TEST_MAX_FILES]
        args.cnn_epochs = config.QUICK_TEST_CNN_EPOCHS
        if args.min_samples == config.MIN_SAMPLES_PER_SENTENCE:
            args.min_samples = 2

    print(f"✓ Will process {len(files)} files")
    print()

    # STEP 2: Build sentence index (only metadata, not data!)
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

    if len(filtered_text_to_files) == 0:
        print("✗ No sentences found with enough samples.")
        return

    # STEP 3: Train/test split (still just metadata!)
    print("STEP 3: Creating Train/Test Split")
    print("-" * 70)

    train_files_list = []
    test_files_list = []

    for text, file_list in filtered_text_to_files.items():
        n = len(file_list)
        n_train = max(1, int(n * args.train_split))
        train_files_list.extend(file_list[:n_train])
        test_files_list.extend(file_list[n_train:])

    print(f"✓ Training files: {len(train_files_list)}")
    print(f"✓ Test files: {len(test_files_list)}")
    print()

    # STEP 4: Train CNN in streaming fashion with SUPERVISED learning
    print("STEP 4: Training CNN Encoder (Supervised, Streaming)")
    print("-" * 70)
    print(f"Processing {len(train_files_list)} files in chunks of {args.chunk_size}")
    print()

    # CHANGED: Create label mapping for supervised training
    unique_sentences = list(filtered_text_to_files.keys())
    sentence_to_label = {sent: idx for idx, sent in enumerate(unique_sentences)}
    num_classes = len(unique_sentences)
    print(f"Number of unique classes: {num_classes}")

    # CHANGED: Initialize SUPERVISED encoder
    encoder = SupervisedCNNEncoder(
        input_channels=config.CNN_INPUT_CHANNELS,
        hidden_channels=config.CNN_HIDDEN_CHANNELS,
        num_classes=num_classes,  # ADDED: Number of sentences
        sequence_length=config.SEQUENCE_LENGTH
    )
    encoder.to(config.CNN_DEVICE)

    # Setup optimizer with lower learning rate for stability
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.cnn_lr * 0.5, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()  # CHANGED: Classification loss instead of MSE
    # Use ReduceLROnPlateau instead of CosineAnnealing for better stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    target_shape = (config.CNN_INPUT_CHANNELS, config.SEQUENCE_LENGTH)

    # Train for multiple epochs
    for epoch in range(args.cnn_epochs):
        print(f"\nEpoch {epoch+1}/{args.cnn_epochs}")
        print("-" * 70)

        epoch_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        # CRITICAL FIX: Shuffle training files before each epoch
        shuffled_train_files = train_files_list.copy()
        random.shuffle(shuffled_train_files)

        # Process in chunks
        for chunk_idx in range(0, len(shuffled_train_files), args.chunk_size):
            chunk_files = shuffled_train_files[chunk_idx:chunk_idx + args.chunk_size]

            # Load this chunk
            chunk_data = []
            chunk_labels = []
            for f in chunk_files:
                data = loader.load_padded_data(f, target_length=config.SEQUENCE_LENGTH)
                text = file_to_text[f]
                label = sentence_to_label[text]

                if data is not None and data.shape == target_shape:
                    # Augment
                    augmented = loader.augment_data(data, num_augmentations=args.num_aug)
                    chunk_data.extend(augmented)
                    chunk_labels.extend([label] * len(augmented))

            if len(chunk_data) == 0:
                continue

            # Train on this chunk
            X_chunk = torch.tensor(np.array(chunk_data), dtype=torch.float32).to(config.CNN_DEVICE)
            y_chunk = torch.tensor(chunk_labels, dtype=torch.long).to(config.CNN_DEVICE)
            dataset = TensorDataset(X_chunk, y_chunk)
            train_loader = TorchDataLoader(dataset, batch_size=args.cnn_batch_size, shuffle=True)

            for batch in train_loader:
                inputs, labels = batch

                # CHANGED: Forward pass returns features and logits
                _, logits = encoder(inputs)
                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                num_batches += 1

            # Free memory
            del X_chunk, y_chunk, dataset, train_loader, chunk_data, chunk_labels
            torch.cuda.empty_cache()

            print(f"  Chunk {chunk_idx//args.chunk_size + 1}/{(len(shuffled_train_files)-1)//args.chunk_size + 1} processed")

        avg_loss = epoch_loss / max(num_batches, 1)
        train_acc = 100. * correct / total if total > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, LR: {current_lr:.6f}")

        # Update learning rate based on accuracy
        scheduler.step(train_acc)

    # Save CNN
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    if args.save_models:
        torch.save({
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, config.CNN_CHECKPOINT_FILE)
        print(f"\n✓ CNN saved to {config.CNN_CHECKPOINT_FILE}")
    print()

    # STEP 5: Extract features for HMM (streaming)
    print("STEP 5: Extracting Features for HMM (Streaming)")
    print("-" * 70)

    encoder.eval()

    # Extract training features in chunks
    print("Extracting training features...")
    hmm_train_list = []
    train_text_list = []

    for chunk_idx in range(0, len(train_files_list), args.chunk_size):
        chunk_files = train_files_list[chunk_idx:chunk_idx + args.chunk_size]

        chunk_data = []
        chunk_texts = []

        for f in chunk_files:
            data = loader.load_padded_data(f, target_length=config.SEQUENCE_LENGTH)
            text = file_to_text[f]

            if data is not None and data.shape == target_shape:
                augmented = loader.augment_data(data, num_augmentations=args.num_aug)
                chunk_data.extend(augmented)
                chunk_texts.extend([text] * len(augmented))

        if len(chunk_data) > 0:
            X_chunk = torch.tensor(np.array(chunk_data), dtype=torch.float32).to(config.CNN_DEVICE)

            with torch.no_grad():
                features = encoder.get_features(X_chunk)
                features_np = features.cpu().numpy()  # FIXED: Move to CPU before numpy

                for j in range(features_np.shape[0]):
                    hmm_train_list.append(features_np[j].T)
                    train_text_list.append(chunk_texts[j])

            del X_chunk, features, features_np, chunk_data
            torch.cuda.empty_cache()

        print(f"  Chunk {chunk_idx//args.chunk_size + 1}/{(len(train_files_list)-1)//args.chunk_size + 1} processed")

    print(f"✓ Extracted {len(hmm_train_list)} training sequences")

    # Extract test features
    print("\nExtracting test features...")
    hmm_test_list = []
    test_text_list = []

    for f in test_files_list:
        data = loader.load_padded_data(f, target_length=config.SEQUENCE_LENGTH)
        text = file_to_text[f]

        if data is not None and data.shape == target_shape:
            X_test = torch.tensor(np.array([data]), dtype=torch.float32).to(config.CNN_DEVICE)

            with torch.no_grad():
                features = encoder.get_features(X_test)
                features_np = features.cpu().numpy()  # FIXED: Move to CPU before numpy
                hmm_test_list.append(features_np[0].T)
                test_text_list.append(text)

            del X_test, features, features_np

    print(f"✓ Extracted {len(hmm_test_list)} test sequences")

    # ADDED: FEATURE NORMALIZATION (CRITICAL!)
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

    # STEP 6: Train HMM
    print("STEP 6: Training HMM Sentence Predictor")
    print("-" * 70)

    predictor = SentencePredictor(
        n_states=args.hmm_states,
        n_features=config.HMM_N_FEATURES
    )
    predictor.train(hmm_train_list, train_text_list, verbose=args.verbose)

    if args.save_models:
        predictor.save(config.HMM_MODELS_FILE)
    print()

    # STEP 7: Evaluate
    print("STEP 7: Evaluating on Test Set")
    print("-" * 70)

    true_labels = []
    pred_labels = []

    for i, X in enumerate(hmm_test_list):
        true_text = test_text_list[i]
        pred_text, score = predictor.predict(X)

        true_labels.append(true_text)
        pred_labels.append(pred_text)

        if args.verbose and i < 10:
            is_correct = (pred_text == true_text)
            print(f"Sample {i+1}:")
            print(f"  True: {true_text[:60]}...")
            print(f"  Pred: {pred_text[:60] if pred_text else 'None'}...")
            print(f"  Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            print()

    overall_acc, sentence_acc = print_evaluation_summary(true_labels, pred_labels, verbose=True)

    # Summary
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
