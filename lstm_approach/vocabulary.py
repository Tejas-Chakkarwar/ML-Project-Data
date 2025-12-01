"""
Vocabulary and tokenization utilities for Seq2Seq model
Handles sentence → word sequence conversion
"""

import pickle
from collections import Counter


class Vocabulary:
    """
    Vocabulary for Seq2Seq model with special tokens.

    Special tokens:
        <PAD>: Padding token (index 0)
        <SOS>: Start of sequence token
        <EOS>: End of sequence token
        <UNK>: Unknown word token
    """

    def __init__(self, min_word_freq=2):
        """
        Initialize vocabulary.

        Args:
            min_word_freq: Minimum frequency for a word to be included
        """
        self.min_word_freq = min_word_freq

        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'

        # Mappings
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

        # Add special tokens first (they get indices 0, 1, 2, 3)
        self._add_word(self.PAD_TOKEN)
        self._add_word(self.SOS_TOKEN)
        self._add_word(self.EOS_TOKEN)
        self._add_word(self.UNK_TOKEN)

    def _add_word(self, word):
        """Add a word to the vocabulary if it doesn't exist."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build_vocab(self, sentences):
        """
        Build vocabulary from list of sentences.

        Args:
            sentences: List of sentence strings
        """
        print("Building vocabulary...")

        # Count word frequencies
        for sentence in sentences:
            words = self.tokenize(sentence)
            self.word_freq.update(words)

        # Add words that meet minimum frequency
        for word, freq in self.word_freq.items():
            if freq >= self.min_word_freq:
                self._add_word(word)

        print(f"✓ Vocabulary built:")
        print(f"  Total words in corpus: {sum(self.word_freq.values()):,}")
        print(f"  Unique words: {len(self.word_freq):,}")
        print(f"  Vocabulary size (freq >= {self.min_word_freq}): {len(self.word2idx):,}")
        print(f"  Most common words: {self.word_freq.most_common(10)}")

    @staticmethod
    def tokenize(sentence):
        """
        Tokenize a sentence into words.
        Simple tokenization: lowercase and split by whitespace.

        Args:
            sentence: String

        Returns:
            List of words
        """
        # Lowercase and split
        words = sentence.lower().strip().split()

        # Remove punctuation from ends of words (keep hyphens inside)
        words = [word.strip('.,!?;:()[]{}"\'-') for word in words]

        # Filter empty strings
        words = [w for w in words if w]

        return words

    def encode(self, sentence, add_sos=True, add_eos=True):
        """
        Convert sentence to sequence of indices.

        Args:
            sentence: String
            add_sos: Whether to add <SOS> token at start
            add_eos: Whether to add <EOS> token at end

        Returns:
            List of indices
        """
        words = self.tokenize(sentence)

        # Convert words to indices
        indices = []

        if add_sos:
            indices.append(self.word2idx[self.SOS_TOKEN])

        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx[self.UNK_TOKEN])

        if add_eos:
            indices.append(self.word2idx[self.EOS_TOKEN])

        return indices

    def decode(self, indices, skip_special_tokens=True):
        """
        Convert sequence of indices back to sentence.

        Args:
            indices: List of indices
            skip_special_tokens: Whether to skip <PAD>, <SOS>, <EOS>

        Returns:
            String (reconstructed sentence)
        """
        words = []
        special_tokens = {self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN}

        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]

                # Stop at <EOS> token
                if word == self.EOS_TOKEN:
                    break

                # Skip special tokens if requested
                if skip_special_tokens and word in special_tokens:
                    continue

                words.append(word)

        return ' '.join(words)

    def __len__(self):
        """Return vocabulary size."""
        return len(self.word2idx)

    @property
    def pad_idx(self):
        """Return index of <PAD> token."""
        return self.word2idx[self.PAD_TOKEN]

    @property
    def sos_idx(self):
        """Return index of <SOS> token."""
        return self.word2idx[self.SOS_TOKEN]

    @property
    def eos_idx(self):
        """Return index of <EOS> token."""
        return self.word2idx[self.EOS_TOKEN]

    @property
    def unk_idx(self):
        """Return index of <UNK> token."""
        return self.word2idx[self.UNK_TOKEN]

    def save(self, filepath):
        """Save vocabulary to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'min_word_freq': self.min_word_freq
            }, f)
        print(f"✓ Vocabulary saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load vocabulary from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        vocab = cls(min_word_freq=data['min_word_freq'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_freq = data['word_freq']

        print(f"✓ Vocabulary loaded from {filepath}")
        print(f"  Vocabulary size: {len(vocab)}")

        return vocab


def test_vocabulary():
    """Test vocabulary functionality."""
    print("Testing Vocabulary...")
    print("=" * 70)

    # Sample sentences
    sentences = [
        "The cat sat on the mat.",
        "The dog ran in the park.",
        "A cat and a dog are friends.",
        "The cat is sleeping.",
        "The dog is running."
    ]

    # Build vocabulary
    vocab = Vocabulary(min_word_freq=1)
    vocab.build_vocab(sentences)

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"PAD index: {vocab.pad_idx}")
    print(f"SOS index: {vocab.sos_idx}")
    print(f"EOS index: {vocab.eos_idx}")

    # Test encoding
    test_sentence = "The cat sat on the mat."
    print(f"\nOriginal: {test_sentence}")

    encoded = vocab.encode(test_sentence)
    print(f"Encoded: {encoded}")

    decoded = vocab.decode(encoded)
    print(f"Decoded: {decoded}")

    # Test with unknown word
    test_sentence2 = "The elephant is huge."
    print(f"\nOriginal: {test_sentence2}")
    encoded2 = vocab.encode(test_sentence2)
    print(f"Encoded: {encoded2}")
    decoded2 = vocab.decode(encoded2)
    print(f"Decoded: {decoded2}")

    print("\n✓ Vocabulary test passed!")


if __name__ == "__main__":
    test_vocabulary()
