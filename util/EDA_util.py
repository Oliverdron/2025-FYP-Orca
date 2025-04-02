"""
Skin Lesion EDA Utility (Minimal Dependencies)
- Uses only sklearn, matplotlib, and your ImageLoader
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from collections import Counter
from img_util import ImageDataLoader

class EDAAnalyzer:
    def __init__(self, data_csv="dataset.csv", image_dir="data/"):
        """Initialize with dataset paths."""
        self.df = pd.read_csv(data_csv)
        self.image_dir = image_dir
        self.image_loader = ImageDataLoader()

    def plot_class_distribution(self, save_path=None):
        """Plot class distribution using matplotlib only."""
        counts = self.df["diagnosis"].value_counts()
        plt.figure(figsize=(10, 5))
        
        # Custom bar plot with percentage labels
        bars = plt.bar(counts.index, counts.values, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        plt.title("Class Distribution")
        plt.xlabel("Diagnosis")
        plt.ylabel("Count")
        
        # Add percentage labels
        total = sum(counts.values)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height} ({100*height/total:.1f}%)',
                    ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def analyze_images(self, sample_size=100):
        """Analyze image properties without progress bars."""
        stats = []
        sample_df = resample(self.df, 
                           n_samples=min(sample_size, len(self.df)),
                           random_state=42)
        
        for _, row in sample_df.iterrows():
            img_path = os.path.join(self.image_dir, row["image_path"])
            img = self.image_loader.load(img_path)
            
            if img is not None:
                stats.append({
                    'width': img.shape[1],
                    'height': img.shape[0],
                    'brightness': np.mean(img)
                })
                print('#', end='', flush=True)  # Simple progress indicator
                
        print()  # Newline after progress
        return pd.DataFrame(stats)

    def generate_report(self, output_dir="result/eda/"):
        """Generate complete EDA report."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Analyzing class distribution...")
        self.plot_class_distribution(
            save_path=os.path.join(output_dir, "class_dist.png"))
        
        print("Analyzing image properties...")
        stats = self.analyze_images()
        stats.describe().to_csv(
            os.path.join(output_dir, "image_stats.csv"))
        
        # Plot resolutions
        plt.figure(figsize=(10, 5))
        plt.scatter(stats['width'], stats['height'], alpha=0.6)
        plt.title("Image Resolutions")
        plt.xlabel("Width (px)")
        plt.ylabel("Height (px)")
        plt.savefig(os.path.join(output_dir, "resolutions.png"))
        plt.close()

        print(f"Report saved to {output_dir}")

if __name__ == "__main__":
    analyzer = EDAAnalyzer(data_csv="../dataset.csv", 
                          image_dir="../data/")
    analyzer.generate_report()