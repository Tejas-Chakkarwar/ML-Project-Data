try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Matplotlib not found. Skipping plotting.")
import numpy as np
from data_loader import DataLoader
import os

def run_eda(data_dir):
    loader = DataLoader(data_dir)
    loader.load_mapping()
    
    # Get the first file
    first_file = loader.get_all_files()[0]
    print(f"Analyzing first file: {first_file}")
    
    data = loader.load_raw_data(first_file)
    
    if data is not None:
        print(f"Data shape: {data.shape}")
        print(f"Data statistics:\nMean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
        print(f"Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
        
        # Plot first few channels
        if PLOT_AVAILABLE:
            plt.figure(figsize=(15, 10))
            for i in range(min(5, data.shape[1])): # Plot first 5 channels
                plt.subplot(5, 1, i+1)
                plt.plot(data[:, i])
                plt.title(f"Channel {i+1}")
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('eda_plot.png')
            print("Saved EDA plot to eda_plot.png")
        else:
            print("Skipping plot generation.")
        
        # Check for NaN or Inf
        if np.isnan(data).any():
            print("WARNING: Data contains NaNs")
        if np.isinf(data).any():
            print("WARNING: Data contains Infs")

if __name__ == "__main__":
    # Assuming running from project root or src
    # Adjust path as needed. The user's data is in 'processed_data'
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processed_data')
    run_eda(data_dir)
