from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import pandas as pd
from pathlib import Path

# Get the project root 
project_root = Path(__file__).parent.parent  

# Load the dataset
df = pd.read_csv(project_root / "annotation_data" / "result_oliver.csv")

ratings_cols = ["Rating_1", "Rating_2", "Rating_3", "Rating_4"]

# Drop rows with any missing ratings (optional, depending on your needs)
df_clean = df.dropna(subset=ratings_cols)

# Convert ratings to integers
ratings = df_clean[ratings_cols].astype(int)

# Determine unique rating categories across all raters
categories = sorted(ratings.stack().unique())

# Create the count matrix
rating_matrix = np.zeros((ratings.shape[0], len(categories)), dtype=int)

# Fill in the matrix
for i, row in ratings.iterrows():
    for j, cat in enumerate(categories):
        rating_matrix[i, j] = (row == cat).sum()

# Convert to a DataFrame for readability (optional)
matrix_df = pd.DataFrame(rating_matrix, columns=[f"Rating={c}" for c in categories])

#getting the kappa score
kappa_score = fleiss_kappa(rating_matrix)
print(f"Kappa Score: {kappa_score}")