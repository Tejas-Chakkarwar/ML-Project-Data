"""
Training script for LSTM-based EEG-to-Text Classification
End-to-end learning without HMM
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
import argparse
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.data_loader import DataLoader
from src.utils import print_evaluation_summary, format_time
import config_lstm as config
from lstm_model import EEGLSTMClassifier, LSTMWithLabelSmoothing, get_model_summary, count_parameters


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_acc, epoch):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LSTM-based EEG-to-Text Classification')

    # Data parameters
    parser.add_argument('--data-dir', type=str, default=config.DATA_DIR,
                       help='Path to processed data directory')
    parser.add_argument('--min-samples', type=int, default=config.MIN_SAMPLES_PER_SENTENCE,
                       help='Minimum samples per sentence')
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation set ratio (rest is test)')

    # Model parameters
    parser.add_argument('--hidden-size', type=int, default=config.LSTM_HIDDEN_SIZE,
                       help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=config.LSTM_NUM_LAYERS,
                       help='Number of LSTM layers')
    parser.add_argument('--attention-heads', type=int, default=config.ATTENTION_HEADS,
                       help='Number of attention heads')
    parser.add_argument('--no-attention', action='store_true',
                       help='Disable attention mechanism')
    parser.add_argument('--no-channel-reduction', action='store_true',
                       help='Disable channel reduction')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=config.WEIGHT_DECAY,
                       help='Weight decay')

    # Augmentation
    parser.add_argument('--num-aug', type=int, default=config.NUM_AUGMENTATIONS,
                       help='Number of augmentations per sample')

    # Other
    parser.add_argument('--device', type=str, default=config.DEVICE,
                       help='Device (cuda or cpu)')
    parser.add_argument('--no-early-stop', action='store_true',
                       help='Disable early stopping')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with reduced data')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training')

    return parser.parse_args()


def load_data(args):
    """
    Load and split data into train/val/test sets.

    Returns:
        train_data, train_labels, train_texts,
        val_data, val_labels, val_texts,
        test_data, test_labels, test_texts,
        sentence_to_label
    """
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

    # Create label mapping
    unique_sentences = sorted(list(filtered_text_to_files.keys()))
    sentence_to_label = {sent: idx for idx, sent in enumerate(unique_sentences)}
    num_classes = len(unique_sentences)

    print(f"Number of classes: {num_classes}")

    # Split data into train/val/test
    train_files, val_files, test_files = [], [], []

    for text, file_list in filtered_text_to_files.items():
        n = len(file_list)
        random.shuffle(file_list)  # Shuffle for randomness

        n_train = max(1, int(n * args.train_split))
        n_val = max(1, int(n * args.val_split))

        train_files.extend(file_list[:n_train])
        val_files.extend(file_list[n_train:n_train + n_val])
        test_files.extend(file_list[n_train + n_val:])

    print(f"\nData split:")
    print(f"  Training: {len(train_files)} files")
    print(f"  Validation: {len(val_files)} files")
    print(f"  Test: {len(test_files)} files")

    # Load and augment data
    def load_files(file_list, augment=False):
        data_list, label_list, text_list = [], [], []

        print(f"Loading {len(file_list)} files...")
        for f in tqdm(file_list):
            data = loader.load_padded_data(f, target_length=config.SEQUENCE_LENGTH)
            text = file_to_text[f]
            label = sentence_to_label[text]

            if data is not None and data.shape == (config.INPUT_CHANNELS, config.SEQUENCE_LENGTH):
                if augment and args.num_aug > 1:
                    augmented = loader.augment_data(data, num_augmentations=args.num_aug)
                    data_list.extend(augmented)
                    label_list.extend([label] * len(augmented))
                    text_list.extend([text] * len(augmented))
                else:
                    data_list.append(data)
                    label_list.append(label)
                    text_list.append(text)

        return np.array(data_list), np.array(label_list), text_list

    print("\nLoading training data (with augmentation)...")
    train_data, train_labels, train_texts = load_files(train_files, augment=True)

    print("\nLoading validation data (no augmentation)...")
    val_data, val_labels, val_texts = load_files(val_files, augment=False)

    print("\nLoading test data (no augmentation)...")
    test_data, test_labels, test_texts = load_files(test_files, augment=False)

    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    return (train_data, train_labels, train_texts,
            val_data, val_labels, val_texts,
            test_data, test_labels, test_texts,
            sentence_to_label)


def train_epoch(model, loader, criterion, optimizer, device, clip_grad_norm):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()

        optimizer.zero_grad()

        # Forward pass
        logits, _ = model(inputs)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()

            logits, _ = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")


def main():
    args = parse_args()
    start_time = time.time()

    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("=" * 70)
    print("LSTM-BASED EEG-TO-TEXT CLASSIFICATION")
    print("=" * 70)
    print()

    # Load data
    (train_data, train_labels, train_texts,
     val_data, val_labels, val_texts,
     test_data, test_labels, test_texts,
     sentence_to_label) = load_data(args)

    num_classes = len(sentence_to_label)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_data, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_data, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )

    train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0, pin_memory=False)
    val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, pin_memory=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)

    # Create model
    print("\n" + "=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)

    model = EEGLSTMClassifier(
        input_channels=config.INPUT_CHANNELS,
        sequence_length=config.SEQUENCE_LENGTH,
        num_classes=num_classes,
        lstm_hidden_size=args.hidden_size,
        lstm_num_layers=args.num_layers,
        lstm_bidirectional=config.LSTM_BIDIRECTIONAL,
        lstm_dropout=config.LSTM_DROPOUT,
        use_attention=not args.no_attention,
        attention_heads=args.attention_heads,
        use_channel_reduction=not args.no_channel_reduction,
        reduced_channels=config.REDUCED_CHANNELS
    )

    get_model_summary(model)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # Loss and optimizer
    if config.USE_LABEL_SMOOTHING:
        criterion = lambda logits, targets: LSTMWithLabelSmoothing(
            nn.Identity(), num_classes, config.LABEL_SMOOTHING
        ).forward(logits.unsqueeze(0), targets.unsqueeze(0))[1]
        # Simple workaround: use CrossEntropyLoss with label smoothing if PyTorch >= 1.10
        criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Learning rate scheduler
    if config.USE_SCHEDULER:
        if config.SCHEDULER_TYPE == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE
            )
    else:
        scheduler = None

    # Early stopping
    if not args.no_early_stop and config.USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=config.EARLY_STOP_PATIENCE,
            min_delta=config.EARLY_STOP_MIN_DELTA
        )
    else:
        early_stopping = None

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config.CLIP_GRAD_NORM
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            if config.SAVE_BEST_ONLY:
                os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'sentence_to_label': sentence_to_label
                }, config.LSTM_CHECKPOINT_FILE)
                print(f"✓ Best model saved (Val Acc: {val_acc:.2f}%)")

        # Learning rate scheduling
        if scheduler is not None:
            if config.SCHEDULER_TYPE == 'cosine':
                scheduler.step()
            else:
                scheduler.step(val_acc)

        # Early stopping
        if early_stopping is not None:
            early_stopping(val_acc, epoch + 1)
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best epoch: {early_stopping.best_epoch}")
                break

    # Load best model for testing
    if config.SAVE_BEST_ONLY and os.path.exists(config.LSTM_CHECKPOINT_FILE):
        checkpoint = torch.load(config.LSTM_CHECKPOINT_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

    # Test evaluation
    print("\n" + "=" * 70)
    print("TESTING")
    print("=" * 70)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Detailed predictions
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()
            logits, _ = model(inputs)
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.numpy())

    # Convert to sentences
    label_to_sentence = {v: k for k, v in sentence_to_label.items()}
    pred_sentences = [label_to_sentence[p] for p in all_preds]
    true_sentences = [label_to_sentence[t] for t in all_true]

    # Compute word-level metrics
    if config.COMPUTE_WORD_LEVEL_METRICS:
        print("\n" + "=" * 70)
        print("WORD-LEVEL EVALUATION")
        print("=" * 70)
        print_evaluation_summary(true_sentences, pred_sentences, verbose=True)

    # Plot training history
    if config.SAVE_TRAINING_PLOT:
        plot_path = os.path.join(config.CHECKPOINT_DIR, 'lstm_training_history.png')
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        plot_training_history(history, save_path=plot_path)

    # Final summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Total Time: {format_time(elapsed_time)}")
    print(f"Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Number of Classes: {num_classes}")
    print(f"Model Parameters: {count_parameters(model):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
