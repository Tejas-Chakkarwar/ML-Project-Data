"""
LSTM-based EEG-to-Text Classification Model with Attention
Provides end-to-end learning without requiring HMM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for focusing on important time steps.

    This helps the model learn which parts of the EEG sequence are most
    relevant for sentence classification.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Linear projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, hidden_size = x.size()

        # Linear projections and reshape for multi-head
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads and apply final linear
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out(context)

        return output, attention_weights


class EEGLSTMClassifier(nn.Module):
    """
    LSTM-based classifier for EEG-to-text classification.

    Architecture:
        Input: (batch, 105, 5500) - Raw EEG signals
        ↓
        Channel Reduction (optional): Conv1d to reduce channels
        ↓
        Transpose: (batch, seq_len, channels) - LSTM expects this format
        ↓
        Bidirectional LSTM: Captures temporal patterns
        ↓
        Multi-head Attention: Focuses on important time steps
        ↓
        Global Pooling: Aggregate temporal information
        ↓
        Fully Connected: Classification head
        ↓
        Output: (batch, num_classes)
    """

    def __init__(
        self,
        input_channels=105,
        sequence_length=5500,
        num_classes=95,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        lstm_bidirectional=True,
        lstm_dropout=0.3,
        use_attention=True,
        attention_heads=4,
        use_channel_reduction=True,
        reduced_channels=32
    ):
        super(EEGLSTMClassifier, self).__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.use_attention = use_attention
        self.use_channel_reduction = use_channel_reduction
        self.lstm_bidirectional = lstm_bidirectional

        # Optional channel reduction using 1D convolution
        if use_channel_reduction:
            self.channel_reducer = nn.Sequential(
                nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(64, reduced_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(reduced_channels),
                nn.ReLU()
            )
            lstm_input_size = reduced_channels
        else:
            lstm_input_size = input_channels

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )

        # Calculate LSTM output size
        lstm_output_size = lstm_hidden_size * 2 if lstm_bidirectional else lstm_hidden_size

        # Multi-head attention (optional)
        if use_attention:
            self.attention = MultiHeadAttention(
                hidden_size=lstm_output_size,
                num_heads=attention_heads,
                dropout=0.1
            )
            self.attention_norm = nn.LayerNorm(lstm_output_size)

        # Global pooling strategies
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Classification head
        # We use both avg and max pooling, so input is 2 * lstm_output_size
        classifier_input_size = lstm_output_size * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'classifier' in name or 'channel_reducer' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, 105, 5500) - Raw EEG signals

        Returns:
            logits: (batch, num_classes) - Classification scores
            attention_weights: (batch, num_heads, seq_len, seq_len) or None
        """
        batch_size = x.size(0)

        # Optional channel reduction
        if self.use_channel_reduction:
            x = self.channel_reducer(x)  # (batch, reduced_channels, 5500)

        # Transpose for LSTM: (batch, seq_len, channels)
        x = x.transpose(1, 2)  # (batch, 5500, channels)

        # Bidirectional LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, lstm_output_size)

        # Optional attention
        attention_weights = None
        if self.use_attention:
            attn_out, attention_weights = self.attention(lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)  # Residual connection

        # Global pooling: combine avg and max pooling
        # Transpose back to (batch, features, seq_len) for pooling
        lstm_out_t = lstm_out.transpose(1, 2)  # (batch, lstm_output_size, seq_len)

        avg_pool = self.avg_pool(lstm_out_t).squeeze(-1)  # (batch, lstm_output_size)
        max_pool = self.max_pool(lstm_out_t).squeeze(-1)  # (batch, lstm_output_size)

        # Concatenate pooling outputs
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch, lstm_output_size * 2)

        # Classification
        logits = self.classifier(pooled)  # (batch, num_classes)

        return logits, attention_weights

    def get_attention_weights(self, x):
        """
        Get attention weights for visualization.
        Useful for understanding which time steps are important.
        """
        with torch.no_grad():
            _, attention_weights = self.forward(x)
        return attention_weights


class LSTMWithLabelSmoothing(nn.Module):
    """
    Wrapper that adds label smoothing to the LSTM classifier.
    Label smoothing helps prevent overconfidence and improves generalization.
    """

    def __init__(self, model, num_classes, smoothing=0.1):
        super(LSTMWithLabelSmoothing, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, targets=None):
        """
        Forward pass with optional label smoothing.

        Args:
            x: Input tensor
            targets: Optional target labels (for computing smoothed loss)

        Returns:
            If targets provided: (logits, smoothed_loss)
            Otherwise: (logits, attention_weights)
        """
        logits, attention_weights = self.model(x)

        if targets is not None:
            # Compute label smoothing loss
            log_probs = F.log_softmax(logits, dim=-1)

            # One-hot encode targets with smoothing
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.smoothing / (self.num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

            loss = (-true_dist * log_probs).sum(dim=-1).mean()
            return logits, loss
        else:
            return logits, attention_weights


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model, input_shape=(105, 5500)):
    """
    Print a summary of the model architecture.

    Args:
        model: The LSTM model
        input_shape: Shape of input (channels, seq_len)
    """
    print("=" * 70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 70)
    print(f"Input shape: (batch, {input_shape[0]}, {input_shape[1]})")
    print(f"Total parameters: {count_parameters(model):,}")
    print()
    print(model)
    print("=" * 70)


if __name__ == "__main__":
    # Test the model
    print("Testing LSTM model...")

    # Create model
    model = EEGLSTMClassifier(
        input_channels=105,
        sequence_length=5500,
        num_classes=95,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        lstm_bidirectional=True,
        use_attention=True,
        attention_heads=4,
        use_channel_reduction=True,
        reduced_channels=32
    )

    # Print summary
    get_model_summary(model)

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 105, 5500)

    print(f"\nTest forward pass with batch_size={batch_size}...")
    logits, attention_weights = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    if attention_weights is not None:
        print(f"Attention weights shape: {attention_weights.shape}")

    print("\n✓ Model test passed!")
