"""
Training script for Seq2Seq EEG-to-Text model
Generates sentences word-by-word from EEG signals
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import argparse
import time
import random
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.data_loader import DataLoader
from src.utils import format_time, calculate_wer
import config_lstm as config
from vocabulary import Vocabulary
from seq2seq_model import Seq2Seq, count_parameters


class EEGTextDataset(Dataset):
    """Dataset for Seq2Seq training."""

    def __init__(self, eeg_data, sentences, vocabulary, max_len=50):
        """
        Args:
            eeg_data: List of EEG arrays (105, 5500)
            sentences: List of sentence strings
            vocabulary: Vocabulary object
            max_len: Maximum sentence length (in words)
        """
        self.eeg_data = eeg_data
        self.sentences = sentences
        self.vocabulary = vocabulary
        self.max_len = max_len

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        sentence = self.sentences[idx]

        # Encode sentence to word indices
        indices = self.vocabulary.encode(sentence, add_sos=True, add_eos=True)

        # Pad or truncate to max_len
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices = indices + [self.vocabulary.pad_idx] * (self.max_len - len(indices))

        return torch.tensor(eeg, dtype=torch.float32), torch.tensor(indices, dtype=torch.long)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Seq2Seq EEG-to-Text Training')

    # Data parameters
    parser.add_argument('--data-dir', type=str, default=config.DATA_DIR,
                       help='Path to processed data directory')
    parser.add_argument('--min-samples', type=int, default=18,
                       help='Minimum samples per sentence (default: 18 for 95 classes)')
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation set ratio')

    # Model parameters
    parser.add_argument('--embedding-dim', type=int, default=256,
                       help='Word embedding dimension')
    parser.add_argument('--encoder-hidden', type=int, default=256,
                       help='Encoder LSTM hidden size')
    parser.add_argument('--decoder-hidden', type=int, default=256,
                       help='Decoder LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--max-len', type=int, default=60,
                       help='Maximum sentence length in words')
    parser.add_argument('--min-word-freq', type=int, default=2,
                       help='Minimum word frequency for vocabulary')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--teacher-forcing', type=float, default=0.5,
                       help='Teacher forcing ratio')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping')

    # Augmentation
    parser.add_argument('--num-aug', type=int, default=6,
                       help='Number of augmentations per sample')

    # Other
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cuda or cpu)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with reduced data')
    parser.add_argument('--no-early-stop', action='store_true',
                       help='Disable early stopping')

    return parser.parse_args()


def load_and_prepare_data(args):
    """Load data and build vocabulary."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    loader = DataLoader(data_dir)
    loader.load_mapping()
    files = loader.get_all_files()

    if args.quick_test:
        print(f"⚡ Quick test mode: using 200 files")
        files = files[:200]

    print(f"Total files: {len(files)}")

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

    # Filter by minimum samples
    filtered_text_to_files = {k: v for k, v in text_to_files.items()
                              if len(v) >= args.min_samples}

    print(f"Unique sentences (>= {args.min_samples} samples): {len(filtered_text_to_files)}")

    if len(filtered_text_to_files) == 0:
        raise ValueError("No sentences found with enough samples!")

    # Build vocabulary from all sentences
    print("\n" + "=" * 70)
    print("BUILDING VOCABULARY")
    print("=" * 70)

    all_sentences = list(filtered_text_to_files.keys())
    vocabulary = Vocabulary(min_word_freq=args.min_word_freq)
    vocabulary.build_vocab(all_sentences)

    # Split data
    print("\n" + "=" * 70)
    print("SPLITTING DATA")
    print("=" * 70)

    train_files, val_files, test_files = [], [], []

    for text, file_list in filtered_text_to_files.items():
        n = len(file_list)
        random.shuffle(file_list)

        n_train = max(1, int(n * args.train_split))
        n_val = max(1, int(n * args.val_split))

        train_files.extend(file_list[:n_train])
        val_files.extend(file_list[n_train:n_train + n_val])
        test_files.extend(file_list[n_train + n_val:])

    print(f"Training: {len(train_files)} files")
    print(f"Validation: {len(val_files)} files")
    print(f"Test: {len(test_files)} files")

    # Load actual data
    def load_files(file_list, augment=False):
        data_list, text_list = [], []

        print(f"Loading {len(file_list)} files...")
        for f in tqdm(file_list):
            data = loader.load_padded_data(f, target_length=config.SEQUENCE_LENGTH)
            text = file_to_text[f]

            if data is not None and data.shape == (config.INPUT_CHANNELS, config.SEQUENCE_LENGTH):
                if augment and args.num_aug > 1:
                    augmented = loader.augment_data(data, num_augmentations=args.num_aug)
                    data_list.extend(augmented)
                    text_list.extend([text] * len(augmented))
                else:
                    data_list.append(data)
                    text_list.append(text)

        return data_list, text_list

    print("\nLoading training data (with augmentation)...")
    train_data, train_texts = load_files(train_files, augment=True)

    print("\nLoading validation data (no augmentation)...")
    val_data, val_texts = load_files(val_files, augment=False)

    print("\nLoading test data (no augmentation)...")
    test_data, test_texts = load_files(test_files, augment=False)

    print(f"\nFinal sizes:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    return (train_data, train_texts, val_data, val_texts,
            test_data, test_texts, vocabulary)


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio, grad_clip):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for eeg, trg in dataloader:
        eeg = eeg.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs, _ = model(eeg, trg, teacher_forcing_ratio=teacher_forcing_ratio)
        # outputs: (batch, max_len, vocab_size)
        # trg: (batch, max_len)

        # Reshape for loss computation
        # outputs: (batch * max_len, vocab_size)
        # trg: (batch * max_len)
        outputs_flat = outputs[:, 1:, :].reshape(-1, outputs.size(2))
        trg_flat = trg[:, 1:].reshape(-1)

        # Compute loss (ignore padding)
        loss = criterion(outputs_flat, trg_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device, vocabulary):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for eeg, trg in dataloader:
            eeg = eeg.to(device)
            trg = trg.to(device)

            # Forward pass with no teacher forcing
            outputs, _ = model(eeg, trg, teacher_forcing_ratio=0)

            # Compute loss
            outputs_flat = outputs[:, 1:, :].reshape(-1, outputs.size(2))
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, trg_flat)

            total_loss += loss.item()
            num_batches += 1

            # Generate predictions
            predictions = outputs.argmax(dim=2)  # (batch, max_len)

            # Decode
            for i in range(predictions.size(0)):
                pred_sent = vocabulary.decode(predictions[i].cpu().numpy())
                true_sent = vocabulary.decode(trg[i].cpu().numpy())

                all_predictions.append(pred_sent)
                all_targets.append(true_sent)

    avg_loss = total_loss / num_batches

    # Compute accuracy metrics
    exact_match = sum([p == t for p, t in zip(all_predictions, all_targets)]) / len(all_predictions)

    # Compute WER (Word Error Rate)
    total_wer = 0
    for pred, true in zip(all_predictions, all_targets):
        wer = calculate_wer(true, pred)
        total_wer += wer
    avg_wer = total_wer / len(all_predictions)

    return avg_loss, exact_match * 100, avg_wer, all_predictions, all_targets


def main():
    args = parse_args()
    start_time = time.time()

    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("SEQ2SEQ EEG-TO-TEXT TRAINING")
    print("=" * 70)
    print()

    # Load data and build vocabulary
    (train_data, train_texts, val_data, val_texts,
     test_data, test_texts, vocabulary) = load_and_prepare_data(args)

    # Create datasets
    train_dataset = EEGTextDataset(train_data, train_texts, vocabulary, max_len=args.max_len)
    val_dataset = EEGTextDataset(val_data, val_texts, vocabulary, max_len=args.max_len)
    test_dataset = EEGTextDataset(test_data, test_texts, vocabulary, max_len=args.max_len)

    # Create data loaders
    train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0)
    val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
    test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)

    # Create model
    print("\n" + "=" * 70)
    print("BUILDING SEQ2SEQ MODEL")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = Seq2Seq(
        vocab_size=len(vocabulary),
        input_channels=config.INPUT_CHANNELS,
        sequence_length=config.SEQUENCE_LENGTH,
        embedding_dim=args.embedding_dim,
        encoder_hidden_size=args.encoder_hidden,
        decoder_hidden_size=args.decoder_hidden,
        num_layers=args.num_layers,
        dropout=0.3,
        use_channel_reduction=True,
        reduced_channels=32
    )

    model = model.to(device)

    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Device: {device}")

    # Loss and optimizer
    # Ignore padding index in loss
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = 10

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            args.teacher_forcing, args.grad_clip
        )

        # Validate
        val_loss, val_acc, val_wer, val_preds, val_targets = evaluate(
            model, val_loader, criterion, device, vocabulary
        )

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val WER: {val_wer:.2f}")

        # Show sample predictions
        if epoch % 5 == 0:
            print("\nSample predictions:")
            for i in range(min(3, len(val_preds))):
                print(f"  True: {val_targets[i]}")
                print(f"  Pred: {val_preds[i]}")
                print()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'vocabulary': vocabulary
            }, os.path.join(config.CHECKPOINT_DIR, 'seq2seq_model.pth'))
            print(f"✓ Best model saved (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if not args.no_early_stop and patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            print(f"Best epoch: {best_epoch}")
            break

    # Load best model for testing
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'seq2seq_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

    # Test evaluation
    print("\n" + "=" * 70)
    print("TESTING")
    print("=" * 70)

    test_loss, test_acc, test_wer, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, vocabulary
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy (Exact Match): {test_acc:.2f}%")
    print(f"Test WER (Word Error Rate): {test_wer:.2f}")

    # Show detailed predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    for i in range(min(10, len(test_preds))):
        print(f"\nExample {i + 1}:")
        print(f"  True: {test_targets[i]}")
        print(f"  Pred: {test_preds[i]}")
        wer = calculate_wer(test_targets[i], test_preds[i])
        print(f"  WER: {wer:.2f}")

    # Final summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Total Time: {format_time(elapsed_time)}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Test Exact Match Accuracy: {test_acc:.2f}%")
    print(f"Test Word Error Rate: {test_wer:.2f}")
    print(f"Vocabulary Size: {len(vocabulary)}")
    print(f"Model Parameters: {count_parameters(model):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
