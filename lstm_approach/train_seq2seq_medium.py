"""
MEDIUM Seq2Seq Training - Best Balance of Memory and Accuracy
Uses gradient accumulation and EEG downsampling to maintain accuracy with less RAM
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
import gc

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.data_loader import DataLoader
from src.utils import format_time, calculate_wer
import config_lstm as config
from vocabulary import Vocabulary
from seq2seq_model import Seq2Seq, count_parameters


class EEGTextDataset(Dataset):
    """Dataset with EEG downsampling to save memory."""

    def __init__(self, eeg_data, sentences, vocabulary, max_len=50, downsample_factor=2):
        """
        Args:
            downsample_factor: Downsample EEG by this factor (2 = half the points)
        """
        self.sentences = sentences
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.downsample_factor = downsample_factor

        # Downsample EEG immediately to save memory
        print(f"  Downsampling EEG by factor of {downsample_factor}...")
        self.eeg_data = []
        for eeg in tqdm(eeg_data, desc="  Processing"):
            # Take every Nth sample (downsampling)
            eeg_downsampled = eeg[:, ::downsample_factor]
            self.eeg_data.append(eeg_downsampled)

        # Free original data
        del eeg_data
        gc.collect()

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        sentence = self.sentences[idx]
        indices = self.vocabulary.encode(sentence, add_sos=True, add_eos=True)

        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices = indices + [self.vocabulary.pad_idx] * (self.max_len - len(indices))

        return torch.tensor(eeg, dtype=torch.float32), torch.tensor(indices, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description='Medium Seq2Seq (Balanced Memory/Accuracy)')

    parser.add_argument('--data-dir', type=str, default=config.DATA_DIR)
    parser.add_argument('--min-samples', type=int, default=18)
    parser.add_argument('--train-split', type=float, default=0.7)
    parser.add_argument('--val-split', type=float, default=0.15)

    # BALANCED PARAMETERS
    parser.add_argument('--embedding-dim', type=int, default=256)  # FULL SIZE
    parser.add_argument('--encoder-hidden', type=int, default=256)  # FULL SIZE
    parser.add_argument('--decoder-hidden', type=int, default=256)  # FULL SIZE
    parser.add_argument('--num-layers', type=int, default=2)  # FULL SIZE
    parser.add_argument('--max-len', type=int, default=50)
    parser.add_argument('--min-word-freq', type=int, default=2)

    # MEMORY OPTIMIZATION TRICKS
    parser.add_argument('--batch-size', type=int, default=4)  # Small physical batch
    parser.add_argument('--accumulation-steps', type=int, default=4)  # Effective batch = 4×4 = 16
    parser.add_argument('--downsample-factor', type=int, default=2)  # EEG downsampling (key!)

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--teacher-forcing', type=float, default=0.5)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    parser.add_argument('--num-aug', type=int, default=6)  # FULL augmentation
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--quick-test', action='store_true')

    return parser.parse_args()


def load_and_prepare_data(args):
    """Load data with streaming approach."""
    print("=" * 70)
    print("LOADING DATA (MEMORY-BALANCED)")
    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    loader = DataLoader(data_dir)
    loader.load_mapping()
    files = loader.get_all_files()

    if args.quick_test:
        files = files[:200]

    print(f"Total files: {len(files)}")

    file_to_text = {}
    text_to_files = {}

    for f in files:
        text = loader.get_text_for_file(f)
        if text:
            file_to_text[f] = text
            if text not in text_to_files:
                text_to_files[text] = []
            text_to_files[text].append(f)

    filtered_text_to_files = {k: v for k, v in text_to_files.items()
                              if len(v) >= args.min_samples}

    print(f"Unique sentences (>= {args.min_samples} samples): {len(filtered_text_to_files)}")

    # Build vocabulary
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

    # Load data
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


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio,
                grad_clip, accumulation_steps):
    """
    Train with GRADIENT ACCUMULATION.
    This simulates larger batch size without holding everything in memory!
    """
    model.train()
    total_loss = 0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, (eeg, trg) in enumerate(dataloader):
        eeg = eeg.to(device)
        trg = trg.to(device)

        # Forward pass
        outputs, _ = model(eeg, trg, teacher_forcing_ratio=teacher_forcing_ratio)

        outputs_flat = outputs[:, 1:, :].reshape(-1, outputs.size(2))
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(outputs_flat, trg_flat)

        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()

        total_loss += loss.item() * accumulation_steps  # Unscale for reporting

        # Only update weights every N steps
        if (batch_idx + 1) % accumulation_steps == 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Memory cleanup
            if num_batches % 20 == 0:
                gc.collect()

        num_batches += 1

        del eeg, trg, outputs, outputs_flat, trg_flat, loss

    # Update any remaining gradients
    if num_batches % accumulation_steps != 0:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device, vocabulary):
    """Evaluate with memory cleanup."""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for eeg, trg in dataloader:
            eeg = eeg.to(device)
            trg = trg.to(device)

            outputs, _ = model(eeg, trg, teacher_forcing_ratio=0)

            outputs_flat = outputs[:, 1:, :].reshape(-1, outputs.size(2))
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, trg_flat)

            total_loss += loss.item()
            num_batches += 1

            predictions = outputs.argmax(dim=2)

            for i in range(predictions.size(0)):
                pred_sent = vocabulary.decode(predictions[i].cpu().numpy())
                true_sent = vocabulary.decode(trg[i].cpu().numpy())
                all_predictions.append(pred_sent)
                all_targets.append(true_sent)

            del eeg, trg, outputs, outputs_flat, trg_flat, predictions
            if num_batches % 10 == 0:
                gc.collect()

    avg_loss = total_loss / num_batches
    exact_match = sum([p == t for p, t in zip(all_predictions, all_targets)]) / len(all_predictions)

    total_wer = 0
    for pred, true in zip(all_predictions, all_targets):
        wer = calculate_wer(true, pred)
        total_wer += wer
    avg_wer = total_wer / len(all_predictions)

    return avg_loss, exact_match * 100, avg_wer, all_predictions, all_targets


def main():
    args = parse_args()
    start_time = time.time()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("SEQ2SEQ MEDIUM - BALANCED VERSION")
    print("=" * 70)
    print("\n⚡ Memory Optimizations:")
    print(f"  ✓ EEG downsampling: {args.downsample_factor}x (5500 → {5500//args.downsample_factor})")
    print(f"  ✓ Gradient accumulation: {args.accumulation_steps} steps")
    print(f"  ✓ Effective batch size: {args.batch_size * args.accumulation_steps}")
    print("\n🎯 Maintains Full Model:")
    print("  ✓ 256 hidden units (not reduced)")
    print("  ✓ 2 LSTM layers (not reduced)")
    print("  ✓ Full augmentation (6x)")
    print()

    # Load data
    (train_data, train_texts, val_data, val_texts,
     test_data, test_texts, vocabulary) = load_and_prepare_data(args)

    # KEY: Adjust sequence length for downsampled EEG
    downsampled_seq_length = config.SEQUENCE_LENGTH // args.downsample_factor

    # Create datasets with downsampling
    print("\n" + "=" * 70)
    print("CREATING DATASETS WITH EEG DOWNSAMPLING")
    print("=" * 70)

    train_dataset = EEGTextDataset(train_data, train_texts, vocabulary,
                                   max_len=args.max_len, downsample_factor=args.downsample_factor)
    val_dataset = EEGTextDataset(val_data, val_texts, vocabulary,
                                 max_len=args.max_len, downsample_factor=args.downsample_factor)
    test_dataset = EEGTextDataset(test_data, test_texts, vocabulary,
                                  max_len=args.max_len, downsample_factor=args.downsample_factor)

    # Data loaders
    train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0, pin_memory=False)
    val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, pin_memory=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)

    # Create FULL-SIZE model
    print("\n" + "=" * 70)
    print("BUILDING FULL-SIZE MODEL")
    print("=" * 70)

    device = torch.device('cpu')

    model = Seq2Seq(
        vocab_size=len(vocabulary),
        input_channels=config.INPUT_CHANNELS,
        sequence_length=downsampled_seq_length,  # Adjusted!
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
    print(f"EEG sequence: {config.SEQUENCE_LENGTH} → {downsampled_seq_length} (downsampled)")
    print(f"Physical batch size: {args.batch_size}")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            args.teacher_forcing, args.grad_clip, args.accumulation_steps
        )

        val_loss, val_acc, val_wer, val_preds, val_targets = evaluate(
            model, val_loader, criterion, device, vocabulary
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val WER: {val_wer:.2f}")

        if epoch % 5 == 0:
            print("\nSample predictions:")
            for i in range(min(3, len(val_preds))):
                print(f"  True: {val_targets[i][:60]}")
                print(f"  Pred: {val_preds[i][:60]}")
                print()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'vocabulary': vocabulary,
                'downsample_factor': args.downsample_factor
            }, os.path.join(config.CHECKPOINT_DIR, 'seq2seq_medium_model.pth'))
            print(f"✓ Best model saved")
        else:
            patience_counter += 1

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"  → LR reduced: {old_lr:.6f} → {new_lr:.6f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

        gc.collect()

    # Test
    print("\n" + "=" * 70)
    print("TESTING")
    print("=" * 70)

    checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, 'seq2seq_medium_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_wer, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, vocabulary
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test WER: {test_wer:.2f}")

    print("\n" + "=" * 70)
    print("SAMPLE TEST PREDICTIONS")
    print("=" * 70)
    for i in range(min(10, len(test_preds))):
        print(f"\n[{i+1}] True: {test_targets[i]}")
        print(f"    Pred: {test_preds[i]}")
        wer = calculate_wer(test_targets[i], test_preds[i])
        print(f"    WER: {wer:.2f}")

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Total Time: {format_time(elapsed_time)}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test WER: {test_wer:.2f}")
    print(f"Model Size: FULL (256 hidden, 2 layers)")
    print(f"Memory Optimization: EEG downsampling + gradient accumulation")
    print("=" * 70)


if __name__ == "__main__":
    main()
