import pandas as pd
import os
import numpy as np

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.mapping_file = os.path.join(data_dir, 'sentence_mapping.csv')
        self.mapping_df = None

    def load_mapping(self):
        """Loads the sentence mapping CSV."""
        if os.path.exists(self.mapping_file):
            self.mapping_df = pd.read_csv(self.mapping_file)
            print(f"Loaded mapping file with {len(self.mapping_df)} entries.")
        else:
            raise FileNotFoundError(f"Mapping file not found at {self.mapping_file}")

    def get_raw_data_path(self, filename):
        """Returns the full path for a raw data file."""
        return os.path.join(self.data_dir, filename)

    def load_raw_data(self, filename):
        """Loads a single raw data CSV file."""
        path = self.get_raw_data_path(filename)
        if os.path.exists(path):
            # Assuming no header or specific header, based on previous view it had rawData1...
            # We'll read it and return as numpy array
            df = pd.read_csv(path)
            return df.values
        else:
            print(f"File {filename} not found.")
            return None

    def get_all_files(self):
        """Returns a list of all raw filenames from the mapping."""
        if self.mapping_df is None:
            self.load_mapping()
        return self.mapping_df['CSVFilename'].tolist()

    def load_padded_data(self, filename, target_length=5500):
        """Loads data and pads/truncates to target_length. Returns (105, target_length)."""
        data = self.load_raw_data(filename)
        if data is None:
            return None
        
        # data is (105, T) based on our finding? 
        # Wait, previous EDA said (105, 4760). 
        # If shape is (105, 4760), then 105 is rows, 4760 is cols.
        # User confirmed 105 is channels.
        # So we pad the second dimension (columns).
        
        n_channels, n_time = data.shape
        
        if n_time > target_length:
            # Truncate
            return data[:, :target_length]
        elif n_time < target_length:
            # Pad with zeros
            pad_width = target_length - n_time
            # Pad columns (axis 1)
            return np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
        else:
            return data

    def get_text_for_file(self, filename):
        """Returns the text content associated with a filename."""
        if self.mapping_df is None:
            self.load_mapping()
        
        # Filter dataframe
        row = self.mapping_df[self.mapping_df['CSVFilename'] == filename]
        if not row.empty:
            return row.iloc[0]['Content']
        return None

    def augment_data(self, data, num_augmentations=5):
        """
        Generates augmented versions of the data with enhanced techniques.
        data: (105, T)
        Returns: List of augmented data arrays (including original).
        """
        augmented = [data]
        n_channels, n_time = data.shape

        for _ in range(num_augmentations):
            # Copy data
            aug = data.copy()

            # 1. Per-channel amplitude scaling (simulates electrode sensitivity variation)
            if np.random.rand() < 0.7:
                scale = np.random.uniform(0.8, 1.2, size=(n_channels, 1))
                aug = aug * scale

            # 2. Channel dropout (simulates bad electrodes or missing data)
            if np.random.rand() < 0.3:
                n_dropout = int(0.05 * n_channels)  # Drop 5% of channels
                dropout_channels = np.random.choice(n_channels, size=n_dropout, replace=False)
                aug[dropout_channels, :] = 0

            # 3. Add Gaussian noise (simulates measurement noise)
            noise_std = np.random.uniform(0.03, 0.08)
            noise = np.random.normal(0, noise_std, size=aug.shape)
            aug = aug + noise

            # 4. Time shifting (simulates different reading speeds)
            if np.random.rand() < 0.5:
                shift = np.random.randint(-50, 50)
                aug = np.roll(aug, shift, axis=1)

            # 5. Temporal smoothing (simulates low-pass filtering variation)
            if np.random.rand() < 0.3:
                # Apply simple moving average
                window = 3
                kernel = np.ones(window) / window
                for ch in range(n_channels):
                    aug[ch, :] = np.convolve(aug[ch, :], kernel, mode='same')

            # 6. Sign flipping for random channels (simulates electrode polarity)
            if np.random.rand() < 0.2:
                n_flip = int(0.1 * n_channels)  # Flip 10% of channels
                flip_channels = np.random.choice(n_channels, size=n_flip, replace=False)
                aug[flip_channels, :] *= -1

            augmented.append(aug)

        return augmented
