import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
import numpy as np
from pathlib import Path

class FeatureAnalysis:
    def __init__(self, data_dir='data', result_dir='result'):
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        os.makedirs(self.result_dir, exist_ok=True)
        self.feature_cols = ['feat_A', 'feat_B', 'feat_C', 
                             'feat_D', 'feat_E', 'feat_F']
        
    def load_dataset(self):
        """Load feature data from dataset.csv"""
        dataset_path = Path("../dataset.csv")  # Will fix this path to non hardcoded later
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
            
        self.df = pd.read_csv(dataset_path)
        
        # Verify we have all expected features
        missing = [f for f in self.feature_cols if f not in self.df.columns]
        if missing:
            print(f"Warning: Missing features {missing} in dataset")
            
        return self.df

    def plot_feature_combinations(self, combinations_to_plot=None):
        """Generate scatter plots with dual coloring scheme"""
        if not hasattr(self, 'df'):
            self.load_dataset()
            
        available_features = [f for f in self.feature_cols if f in self.df.columns]
        
        if combinations_to_plot is None:
            combinations_to_plot = list(permutations(available_features, 2))  # Use permutations to get A vs B and B vs A
            
        scatter_dir = self.result_dir / "scatter_plots"
        os.makedirs(scatter_dir, exist_ok=True)
        
        for feat1, feat2 in combinations_to_plot:
            plt.figure(figsize=(12, 8))

            # Calculate axis limits using percentiles (focus on middle 95% of data)
            x_min = np.percentile(self.df[feat1], 2.5)
            x_max = np.percentile(self.df[feat1], 97.5)
            y_min = np.percentile(self.df[feat2], 2.5)
            y_max = np.percentile(self.df[feat2], 97.5)

            # Create base scatter plot (feature-based coloring)
            plt.scatter(self.df[feat1], self.df[feat2], 
                        c='blue', alpha=0.8, label='Non-cancerous', s=20)

            # Add cancer outlines if label_binary exists
            if 'label_binary' in self.df.columns:
                cancer_mask = self.df['label_binary'] == True
                plt.scatter(self.df[feat1][cancer_mask], self.df[feat2][cancer_mask],
                            facecolors='none', edgecolors='red', 
                            linewidths=1, s=50, alpha=0.8,
                            label='Cancerous')

            plt.title(f'{feat1} (X) vs {feat2} (Y)\nRed outline = Cancerous')
            plt.xlabel(f'{feat1} Feature Value')
            plt.ylabel(f'{feat2} Feature Value')

            plt.xlim(x_min - 0.05*(x_max - x_min), x_max + 0.05*(x_max - x_min))
            plt.ylim(y_min - 0.05*(y_max - y_min), y_max + 0.05*(y_max - y_min))

            # Improve legend
            handles, labels = plt.gca().get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))  # Remove duplicates
            plt.legend(unique_labels.values(), unique_labels.keys(),
                       bbox_to_anchor=(1.05, 1), loc='upper left')

            # Grid and layout
            plt.grid(True, linestyle='--', alpha=0.3)

            # Add correlation coefficient
            corr = self.df[[feat1, feat2]].corr().iloc[0, 1]
            plt.text(0.05, 0.95, f'Pearson r = {corr:.2f}', 
                     transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout()

            # Save the plot
            plot_path = scatter_dir / f'{feat1}_vs_{feat2}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'Saved enhanced scatter plot: {plot_path}')    
              
    def analyze_all_features(self):
        """Run complete analysis pipeline"""
        self.load_dataset()
        self.plot_feature_combinations()
        
        # Create correlation matrix
        available_features = [f for f in self.feature_cols if f in self.df.columns]
        if len(available_features) > 1:
            corr_matrix = self.df[available_features].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.result_dir / 'feature_correlation.png', dpi=300)
            plt.close()
        
        # Create feature distributions
        self.plot_feature_distributions()
        
        return self.df.describe()
    
    def plot_feature_distributions(self):
        """Plot distributions of all features"""
        available_features = [f for f in self.feature_cols if f in self.df.columns]
        
        dist_dir = self.result_dir / "feature_distributions"
        os.makedirs(dist_dir, exist_ok=True)
        
        for feature in available_features:
            plt.figure(figsize=(8, 5))
            
            # Plot based on label if available
            if 'label_binary' in self.df.columns:
                sns.histplot(data=self.df, x=feature, hue='label_binary', 
                             element='step', stat='density', common_norm=False)
            else:
                sns.histplot(data=self.df, x=feature, kde=True)
                
            plt.title(f'Distribution of {feature}')
            plt.tight_layout()
            plt.savefig(dist_dir / f'{feature}_distribution.png', dpi=300)
            plt.close()
            print(f'Saved distribution plot: {dist_dir / f"{feature}_distribution.png"}')
