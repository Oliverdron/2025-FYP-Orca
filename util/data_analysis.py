'''1) Metadata Summary + Visualization
This script provides a summary and visualization of the metadata from the skin cancer dataset.

DATASET SUMMARY:
Total entries in data set: 2298
Non-null counts for key features: 1494 *for all columns

AGE DISTRIBUTION:
Our model is mainly trained on adult to elderly patients,(extrapolating to younger patients may not be accurate)
    - Median age of patients in the dataset is 60 years
    - Age range: 6 to 94

GENDER DISTRIBUTION:
Slight imbalance in gender distribution, with a higher number of females.
    - Female: 753 
    - Male: 741

CANCER HISTORY:
Patients with no past history of skin cancer: 813
Most common cancer diagnoses:
    - BCC: 845
    - ACK: 730
    - NEV: 244

# '''
# 1) Dataset summary analysis:
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Metadata Summary
# # Load dataset
project_root = Path(__file__).parent.parent 
metadata_path = project_root / "data" / "metadata.csv"
df = pd.read_csv(metadata_path)

#DATASET SUMMARY 
# Total number of entries
total_entries = len(df)
# print(f"# 2) Total entries in dataset: {total_entries}")

# Non-null counts for key partially-filled columns
key_cols = ['gender', 'drink', 'smoke', 'pesticide', 'skin_cancer_history']
non_null_counts = df[key_cols].count()
nocancerhistory = df['skin_cancer_history']
number_nocancerhistory = nocancerhistory[nocancerhistory == False].count()
# print(f' No past history of cancer - Number of patients: {number_nocancerhistory}')
# print("    - Non-null counts for key features:")
# for col, count in non_null_counts.items():
#     print(f"        {col}: {count} entries")

#AGE DISTRIBUTION
median_age = df['age'].median()
age_min = df['age'].min()
age_max = df['age'].max()
# print(f'Median age of patients in the dataset: {median_age} years')
# print(f'Age range: {age_min} to {age_max} years')

# GENDER DISTRUBUTION
gender_counts = df['gender'].value_counts()
# print(f"# 3) Gender distribution:")
# for gender, count in gender_counts.items():
#     print(f"{gender}: {count}")

# Skin cancer history
skin_cancer_counts = df['skin_cancer_history'].value_counts()
# print(f"4) Skin cancer history:")
# for hist, count in skin_cancer_counts.items():
#     print(f"{hist}: {count}")

# Most common diagnoses
diagnosis_counts = df['diagnostic'].value_counts().head(3)
# print("5) Most common diagnoses:")
# for diag, count in diagnosis_counts.items():
#     print(f"{diag}: {count}")


##2) DATA SET VISUALIZATION 



# # Plot histograms for numeric columns
# numeric_cols = df.select_dtypes(include=['number']).columns.drop('lesion_id', errors='ignore')

# for col in numeric_cols:
#     plt.figure()
#     df[col].hist()
#     plt.title(f'Distribution of {col}')
#     plt.xlabel(col)
#     plt.ylabel('Frequency')
#     plt.tight_layout()
#     plt.show()

# # Compute correlation matrix, excluding specified non-meaningful columns
# numeric_df = df.select_dtypes(include=['number']).drop(['biopsed', 'lesion_id'], axis=1, errors='ignore')
# corr = numeric_df.corr()

# # Plot correlation matrix
# plt.figure()
# plt.imshow(corr, aspect='auto')
# plt.title('Correlation Matrix (excluding biopsed and lesion_id)')
# plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
# plt.yticks(range(len(corr.index)), corr.index)
# plt.colorbar()
# plt.tight_layout()
# plt.show()
