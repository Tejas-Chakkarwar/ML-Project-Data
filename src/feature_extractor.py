import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import torch.nn.functional as F

class CNNEEGEncoder(nn.Module):
    def __init__(self, input_channels=105, hidden_channels=32, sequence_length=5500):
        super(CNNEEGEncoder, self).__init__()

        # Encoder: Compresses temporal dimension and extracts features
        # Input: (Batch, 105, 5500)
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            # Layer 2
            nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            # Layer 3
            nn.Conv1d(32, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )

        # Decoder: Reconstructs original signal
        self.decoder = nn.Sequential(
            # Layer 1
            nn.ConvTranspose1d(hidden_channels, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            # Layer 2
            nn.ConvTranspose1d(32, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            # Layer 3
            nn.ConvTranspose1d(64, input_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            # No activation for final output (reconstruction)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Ensure output size matches input size
        if decoded.shape[2] != x.shape[2]:
            decoded = F.interpolate(decoded, size=x.shape[2], mode='linear', align_corners=False)

        return encoded, decoded

    def get_features(self, x):
        """Returns the encoded features."""
        with torch.no_grad():
            return self.encoder(x)


class SupervisedCNNEncoder(nn.Module):
    """
    Supervised CNN encoder that learns discriminative features for sentence classification.
    Uses classification loss instead of reconstruction loss for better feature quality.
    """
    def __init__(self, input_channels=105, hidden_channels=32, num_classes=344, sequence_length=5500):
        super(SupervisedCNNEncoder, self).__init__()

        # Same encoder architecture as autoencoder
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            # Layer 2
            nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            # Layer 3
            nn.Conv1d(32, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )

        # Classification head (only used during training)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for training.
        Returns both features and classification logits.
        """
        features = self.encoder(x)  # (batch, 32, ~688)
        pooled = self.pool(features).squeeze(-1)  # (batch, 32)
        logits = self.classifier(pooled)  # (batch, num_classes)
        return features, logits

    def get_features(self, x):
        """Returns the encoded features for HMM."""
        with torch.no_grad():
            return self.encoder(x)

def train_autoencoder(model, data_loader, num_epochs=10, learning_rate=1e-3, 
                      device='cpu', checkpoint_path=None, val_loader=None):
    """
    Train the CNN autoencoder with optional checkpointing and validation
    
    Args:
        model: CNNEEGEncoder model
        data_loader: Training data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        checkpoint_path: Optional path to save best model checkpoint
        val_loader: Optional validation data loader
        
    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    print("Starting Autoencoder training...")
    print(f"Device: {device}, Epochs: {num_epochs}, Learning Rate: {learning_rate}")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in data_loader:
            inputs = batch[0].to(device).float()
            
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_train_loss = total_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase (if validation loader provided)
        val_loss_str = ""
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0].to(device).float()
                    _, outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            history['val_loss'].append(avg_val_loss)
            val_loss_str = f", Val Loss: {avg_val_loss:.4f}"
            
            # Save best model
            if checkpoint_path and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, checkpoint_path)
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}{val_loss_str} ✓ (Best)")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}{val_loss_str}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
            
            # Save checkpoint even without validation
            if checkpoint_path and epoch == num_epochs - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                }, checkpoint_path)
    
    print("-" * 70)
    if checkpoint_path:
        print(f"✓ Model checkpoint saved to {checkpoint_path}")

    return model, history


def train_supervised_encoder(model, data_loader, num_epochs=5, learning_rate=1e-3,
                            device='cpu', checkpoint_path=None):
    """
    Train the supervised CNN encoder with classification loss.

    Args:
        model: SupervisedCNNEncoder model
        data_loader: Training data loader (must include labels)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        checkpoint_path: Optional path to save best model checkpoint

    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    model.to(device)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': []
    }

    best_acc = 0.0

    print("Starting Supervised CNN training...")
    print(f"Device: {device}, Epochs: {num_epochs}, Learning Rate: {learning_rate}")
    print("-" * 70)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        for batch in data_loader:
            inputs = batch[0].to(device).float()
            labels = batch[1].to(device).long()

            optimizer.zero_grad()
            _, logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            num_batches += 1

        scheduler.step()

        avg_train_loss = total_loss / num_batches
        train_acc = 100. * correct / total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Save best model
        if checkpoint_path and train_acc > best_acc:
            best_acc = train_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
            }, checkpoint_path)

    print("-" * 70)
    if checkpoint_path:
        print(f"✓ Model checkpoint saved to {checkpoint_path}")
        print(f"✓ Best training accuracy: {best_acc:.2f}%")

    return model, history
