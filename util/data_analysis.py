import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
project_root = Path(__file__).parent.parent 
metadata_path = project_root / "data" / "metadata.csv"
df = pd.read_csv(metadata_path)


# Plot histograms for numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
for col in numeric_cols:
    plt.figure()
    df[col].hist()
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Correlation matrix
corr = df.corr()
plt.figure()
plt.imshow(corr, aspect='auto')
plt.title('Correlation Matrix')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar()
plt.tight_layout()
plt.show()
