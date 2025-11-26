import numpy as np
import pickle
import os
from hmm_model import GaussianHMM

class SentencePredictor:
    def __init__(self, n_states=3, n_features=32):
        self.n_states = n_states
        self.n_features = n_features
        # Dictionary to store HMMs: {text_content: GaussianHMM}
        self.models = {}
        
    def train(self, X_list, text_list, verbose=True):
        """
        Trains HMMs for each unique sentence.
        
        Args:
            X_list: List of feature sequences (T, n_features)
            text_list: List of corresponding text strings
            verbose: If True, print detailed training progress
        """
        # Group data by text
        grouped_data = {}
        for X, text in zip(X_list, text_list):
            if text not in grouped_data:
                grouped_data[text] = []
            grouped_data[text].append(X)
            
        print(f"\nTraining {len(grouped_data)} distinct sentence models...")
        print("-" * 70)
        
        successful = 0
        failed = 0
        
        for idx, (text, data_list) in enumerate(grouped_data.items(), 1):
            if verbose:
                display_text = text[:50] + "..." if len(text) > 50 else text
                print(f"[{idx}/{len(grouped_data)}] Training: '{display_text}' ({len(data_list)} samples)")
            
            hmm = GaussianHMM(n_states=self.n_states, n_features=self.n_features, 
                            max_iter=10, tol=1e-4)
            
            # Train on all samples available for this sentence
            try:
                hmm.train(data_list)
                self.models[text] = hmm
                successful += 1
            except Exception as e:
                failed += 1
                if verbose:
                    print(f"  ✗ Failed: {e}")
        
        print("-" * 70)
        print(f"✓ Successfully trained {successful} models")
        if failed > 0:
            print(f"✗ Failed to train {failed} models")
        print()
                
    def predict(self, X, top_k=1):
        """
        Predicts the text for a given sequence X by finding the HMM with max likelihood.
        
        Args:
            X: Feature sequence (T, n_features)
            top_k: Return top k predictions
            
        Returns:
            If top_k == 1: (predicted_text, score)
            If top_k > 1: List of (text, score) tuples sorted by score
        """
        if len(self.models) == 0:
            raise ValueError("No models trained. Call train() first.")
        
        scores = []
        
        for text, hmm in self.models.items():
            try:
                score = hmm.score(X)
                scores.append((text, score))
            except Exception as e:
                # If scoring fails, assign very low score
                scores.append((text, -float('inf')))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k == 1:
            return scores[0] if scores else (None, -float('inf'))
        else:
            return scores[:top_k]
    
    def save(self, filepath):
        """
        Save the trained models to a file
        
        Args:
            filepath: Path to save the models
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'n_states': self.n_states,
            'n_features': self.n_features,
            'models': self.models
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✓ Saved {len(self.models)} HMM models to {filepath}")
    
    def load(self, filepath):
        """
        Load trained models from a file
        
        Args:
            filepath: Path to the saved models
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.n_states = save_dict['n_states']
        self.n_features = save_dict['n_features']
        self.models = save_dict['models']
        
        print(f"✓ Loaded {len(self.models)} HMM models from {filepath}")
