#our model is mainly trained on middle aged patients


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
project_root = Path(__file__).parent.parent 
metadata_path = project_root / "data" / "metadata.csv"
df = pd.read_csv(metadata_path)


# Plot histograms for numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns.drop('lesion_id', errors='ignore')

for col in numeric_cols:
    plt.figure()
    df[col].hist()
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Compute correlation matrix, excluding specified non-meaningful columns
numeric_df = df.select_dtypes(include=['number']).drop(['biopsed', 'lesion_id'], axis=1, errors='ignore')
corr = numeric_df.corr()

# Plot correlation matrix
plt.figure()
plt.imshow(corr, aspect='auto')
plt.title('Correlation Matrix (excluding biopsed and lesion_id)')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar()
plt.tight_layout()
plt.show()
