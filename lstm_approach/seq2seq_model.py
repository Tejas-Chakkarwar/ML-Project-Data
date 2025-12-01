"""
Sequence-to-Sequence LSTM model with Attention for EEG-to-Text
Encoder-Decoder architecture with Bahdanau attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    """
    Encoder: Processes EEG signal and produces context vectors.

    Input: EEG signal (batch, 105, 5500)
    Output: Hidden states (batch, seq_len, hidden_size * 2)
    """

    def __init__(
        self,
        input_channels=105,
        sequence_length=5500,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        use_channel_reduction=True,
        reduced_channels=32
    ):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_channel_reduction = use_channel_reduction

        # Optional channel reduction
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
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, 105, 5500)

        Returns:
            outputs: (batch, seq_len, hidden_size * 2)
            hidden: tuple of (h_n, c_n) for LSTM
        """
        # Channel reduction
        if self.use_channel_reduction:
            x = self.channel_reducer(x)  # (batch, reduced_channels, 5500)

        # Transpose for LSTM
        x = x.transpose(1, 2)  # (batch, 5500, channels)

        # LSTM
        outputs, (h_n, c_n) = self.lstm(x)

        # outputs: (batch, 5500, hidden_size * 2)
        # h_n: (num_layers * 2, batch, hidden_size)
        # c_n: (num_layers * 2, batch, hidden_size)

        return outputs, (h_n, c_n)


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.

    Computes attention weights to focus on relevant parts of encoder output.
    """

    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(BahdanauAttention, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        # Attention layers
        self.W_encoder = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.W_decoder = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.V = nn.Linear(decoder_hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Compute attention weights and context vector.

        Args:
            decoder_hidden: (batch, decoder_hidden_size)
            encoder_outputs: (batch, seq_len, encoder_hidden_size)

        Returns:
            context: (batch, encoder_hidden_size)
            attention_weights: (batch, seq_len)
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # Expand decoder hidden to match encoder sequence length
        # decoder_hidden: (batch, decoder_hidden_size) → (batch, seq_len, decoder_hidden_size)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, seq_len, -1)

        # Compute attention scores
        # encoder_outputs: (batch, seq_len, encoder_hidden_size) → (batch, seq_len, decoder_hidden_size)
        encoder_transformed = self.W_encoder(encoder_outputs)

        # decoder_hidden_expanded: (batch, seq_len, decoder_hidden_size)
        decoder_transformed = self.W_decoder(decoder_hidden_expanded)

        # Additive attention
        # (batch, seq_len, decoder_hidden_size)
        combined = torch.tanh(encoder_transformed + decoder_transformed)

        # (batch, seq_len, 1) → (batch, seq_len)
        scores = self.V(combined).squeeze(2)

        # Attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Context vector: weighted sum of encoder outputs
        # attention_weights: (batch, seq_len) → (batch, 1, seq_len)
        # encoder_outputs: (batch, seq_len, encoder_hidden_size)
        # context: (batch, 1, encoder_hidden_size) → (batch, encoder_hidden_size)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class Decoder(nn.Module):
    """
    Decoder: Generates words one at a time using attention over encoder outputs.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_size=256,
        encoder_hidden_size=512,  # 256 * 2 (bidirectional)
        num_layers=2,
        dropout=0.3
    ):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Attention
        self.attention = BahdanauAttention(encoder_hidden_size, hidden_size)

        # LSTM (input = embedding + context)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size + encoder_hidden_size + embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, input_word, decoder_hidden, decoder_cell, encoder_outputs):
        """
        Forward pass for one time step.

        Args:
            input_word: (batch,) - word indices
            decoder_hidden: (num_layers, batch, hidden_size)
            decoder_cell: (num_layers, batch, hidden_size)
            encoder_outputs: (batch, seq_len, encoder_hidden_size)

        Returns:
            output: (batch, vocab_size) - word predictions
            decoder_hidden: updated hidden state
            decoder_cell: updated cell state
            attention_weights: (batch, seq_len)
        """
        batch_size = input_word.size(0)

        # Embed input word: (batch,) → (batch, embedding_dim)
        embedded = self.embedding(input_word)

        # Compute attention using top layer hidden state
        # decoder_hidden: (num_layers, batch, hidden_size)
        # Use last layer: (batch, hidden_size)
        decoder_hidden_top = decoder_hidden[-1]

        # Get context and attention weights
        context, attention_weights = self.attention(decoder_hidden_top, encoder_outputs)
        # context: (batch, encoder_hidden_size)
        # attention_weights: (batch, seq_len)

        # Concatenate embedding and context
        lstm_input = torch.cat([embedded, context], dim=1)  # (batch, embedding_dim + encoder_hidden_size)
        lstm_input = lstm_input.unsqueeze(1)  # (batch, 1, embedding_dim + encoder_hidden_size)

        # LSTM step
        lstm_output, (decoder_hidden, decoder_cell) = self.lstm(lstm_input, (decoder_hidden, decoder_cell))
        # lstm_output: (batch, 1, hidden_size)

        lstm_output = lstm_output.squeeze(1)  # (batch, hidden_size)

        # Concatenate LSTM output, context, and embedding for final prediction
        combined = torch.cat([lstm_output, context, embedded], dim=1)
        # (batch, hidden_size + encoder_hidden_size + embedding_dim)

        # Output layer
        output = self.fc_out(combined)  # (batch, vocab_size)

        return output, decoder_hidden, decoder_cell, attention_weights


class Seq2Seq(nn.Module):
    """
    Complete Sequence-to-Sequence model.

    Combines Encoder and Decoder with attention.
    """

    def __init__(
        self,
        vocab_size,
        input_channels=105,
        sequence_length=5500,
        embedding_dim=256,
        encoder_hidden_size=256,
        decoder_hidden_size=256,
        num_layers=2,
        dropout=0.3,
        use_channel_reduction=True,
        reduced_channels=32
    ):
        super(Seq2Seq, self).__init__()

        self.vocab_size = vocab_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = Encoder(
            input_channels=input_channels,
            sequence_length=sequence_length,
            hidden_size=encoder_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_channel_reduction=use_channel_reduction,
            reduced_channels=reduced_channels
        )

        # Decoder
        encoder_output_size = encoder_hidden_size * 2  # Bidirectional
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=decoder_hidden_size,
            encoder_hidden_size=encoder_output_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Transform encoder hidden state to decoder hidden state
        # Encoder is bidirectional, so we need to combine forward and backward
        self.encoder_to_decoder_h = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)
        self.encoder_to_decoder_c = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass with teacher forcing.

        Args:
            src: (batch, 105, 5500) - EEG signals
            trg: (batch, max_len) - target word indices
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            outputs: (batch, max_len, vocab_size)
            all_attention_weights: list of attention weights for each step
        """
        batch_size = src.size(0)
        max_len = trg.size(1)

        # Encode
        encoder_outputs, (h_n, c_n) = self.encoder(src)
        # encoder_outputs: (batch, 5500, encoder_hidden_size * 2)
        # h_n: (num_layers * 2, batch, encoder_hidden_size)
        # c_n: (num_layers * 2, batch, encoder_hidden_size)

        # Initialize decoder hidden state
        # Combine forward and backward of last layer
        # h_n shape: (num_layers * 2, batch, encoder_hidden_size)
        # Reshape to (num_layers, 2, batch, encoder_hidden_size)
        h_n = h_n.view(self.num_layers, 2, batch_size, self.encoder_hidden_size)
        c_n = c_n.view(self.num_layers, 2, batch_size, self.encoder_hidden_size)

        # Concatenate forward and backward for each layer
        h_n_combined = torch.cat([h_n[:, 0, :, :], h_n[:, 1, :, :]], dim=2)  # (num_layers, batch, hidden*2)
        c_n_combined = torch.cat([c_n[:, 0, :, :], c_n[:, 1, :, :]], dim=2)

        # Transform to decoder hidden size
        decoder_hidden = self.encoder_to_decoder_h(h_n_combined)  # (num_layers, batch, decoder_hidden)
        decoder_cell = self.encoder_to_decoder_c(c_n_combined)

        # Storage for outputs
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(src.device)
        all_attention_weights = []

        # First input is <SOS> token
        input_word = trg[:, 0]

        # Decode
        for t in range(1, max_len):
            output, decoder_hidden, decoder_cell, attention_weights = self.decoder(
                input_word, decoder_hidden, decoder_cell, encoder_outputs
            )

            outputs[:, t, :] = output
            all_attention_weights.append(attention_weights)

            # Teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing:
                input_word = trg[:, t]
            else:
                input_word = output.argmax(1)

        return outputs, all_attention_weights

    def generate(self, src, max_len=50, sos_idx=1, eos_idx=2):
        """
        Generate sentence without teacher forcing (inference mode).

        Args:
            src: (batch, 105, 5500)
            max_len: Maximum length to generate
            sos_idx: Index of <SOS> token
            eos_idx: Index of <EOS> token

        Returns:
            generated: (batch, max_len) - generated word indices
            all_attention_weights: list of attention weights
        """
        batch_size = src.size(0)

        # Encode
        encoder_outputs, (h_n, c_n) = self.encoder(src)

        # Initialize decoder hidden state
        h_n = h_n.view(self.num_layers, 2, batch_size, self.encoder_hidden_size)
        c_n = c_n.view(self.num_layers, 2, batch_size, self.encoder_hidden_size)

        h_n_combined = torch.cat([h_n[:, 0, :, :], h_n[:, 1, :, :]], dim=2)
        c_n_combined = torch.cat([c_n[:, 0, :, :], c_n[:, 1, :, :]], dim=2)

        decoder_hidden = self.encoder_to_decoder_h(h_n_combined)
        decoder_cell = self.encoder_to_decoder_c(c_n_combined)

        # Start with <SOS>
        input_word = torch.LongTensor([sos_idx] * batch_size).to(src.device)

        generated = []
        all_attention_weights = []

        for t in range(max_len):
            output, decoder_hidden, decoder_cell, attention_weights = self.decoder(
                input_word, decoder_hidden, decoder_cell, encoder_outputs
            )

            # Get predicted word
            predicted_word = output.argmax(1)
            generated.append(predicted_word.unsqueeze(1))
            all_attention_weights.append(attention_weights)

            # Next input is predicted word
            input_word = predicted_word

            # Stop if all sequences generated <EOS>
            if (predicted_word == eos_idx).all():
                break

        # Concatenate generated words
        generated = torch.cat(generated, dim=1)  # (batch, actual_len)

        return generated, all_attention_weights


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Seq2Seq model...")
    print("=" * 70)

    # Create model
    vocab_size = 1000
    model = Seq2Seq(
        vocab_size=vocab_size,
        input_channels=105,
        sequence_length=5500,
        embedding_dim=256,
        encoder_hidden_size=128,
        decoder_hidden_size=128,
        num_layers=2,
        dropout=0.3
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    max_len = 20

    src = torch.randn(batch_size, 105, 5500)
    trg = torch.randint(0, vocab_size, (batch_size, max_len))

    print(f"\nTesting forward pass (training mode)...")
    print(f"  Input shape: {src.shape}")
    print(f"  Target shape: {trg.shape}")

    outputs, attention_weights = model(src, trg, teacher_forcing_ratio=0.5)
    print(f"  Output shape: {outputs.shape}")
    print(f"  Attention weights: {len(attention_weights)} steps")

    print(f"\nTesting generation (inference mode)...")
    generated, attention_weights = model.generate(src, max_len=20, sos_idx=1, eos_idx=2)
    print(f"  Generated shape: {generated.shape}")

    print("\n✓ Seq2Seq model test passed!")
