import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy.stats import chi2_contingency

''' Metadata Summary + Visualization + Further Exploration
==========================================================
This file provides a summary and visualization of the metadata from the skin cancer dataset.
1) Metadata Summary:
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
    -BCC: 845
    -ACK: 730
    -NEV: 244
    -SEK: 235
    -SCC: 192
    -MEL: 52

2) DataSet Visualization:

From the FitsPatrick Histogram Visualisation- the data is primary from the Fitzpatrick type 1 to 3 skin types
with a very few cases from Fitzpatrick type 4 to 6. Meaning that the dataset is primarily consisting of patients with lighter skin tones.
This will affect the model's performance on patients with darker skin tones. With not much darker skin tone data to train the model, 
this is a limitation of the dataset and should be considered when evaluating the model's validity and credibility. 

3) Further exploration - if we want to explore and search for variables that might be causing a patient to have skin cancer- we can experiment with 
using contingency tables and chi-squared tests to analyze the relationship between binary features and the presence of cancer.
for example, in this case looking at variables like smoking, drinking, skin cancer history and comparing them to the presence of cancer.
we see that smoking's Chi-squared test result show: χ² = 10.98, p = 0.0009,(p < 0.05) suggesting that there is a significant relationship between smoking 
and the presence of skin cancer. This is maybe something in the future our model can take into account when predicting if there is cancer or even what type of cancer.
This may result in an increase of the model's accuracy and validity.
==========================================================

# '''
# 1) Dataset summary analysis: 
# Metadata Summary
project_root = Path(__file__).parent.parent 
metadata_path = project_root / "data" / "metadata.csv"
df = pd.read_csv(metadata_path)

#DATASET SUMMARY 
total_entries = len(df) # Total number of entries
print(f"# 2) Total entries in dataset: {total_entries}")


key_cols = ['gender', 'drink', 'smoke', 'pesticide', 'skin_cancer_history'] # Non-null counts for key partially-filled columns
non_null_counts = df[key_cols].count()
nocancerhistory = df['skin_cancer_history']
number_nocancerhistory = nocancerhistory[nocancerhistory == False].count()
print(f' No past history of cancer - Number of patients: {number_nocancerhistory}')
print("    - Non-null counts for key features:")
for col, count in non_null_counts.items():
    print(f"        {col}: {count} entries")

#AGE DISTRIBUTION
median_age = df['age'].median()
age_min = df['age'].min()
age_max = df['age'].max()
print(f'Median age of patients in the dataset: {median_age} years')
print(f'Age range: {age_min} to {age_max} years')

# GENDER DISTRUBUTION
gender_counts = df['gender'].value_counts()
print(f"# 3) Gender distribution:")
for gender, count in gender_counts.items():
    print(f"{gender}: {count}")

# Skin cancer history
skin_cancer_counts = df['skin_cancer_history'].value_counts()
print(f"4) Skin cancer history:")
for hist, count in skin_cancer_counts.items():
    print(f"{hist}: {count}")

# Valuecounts for type of cancers
diagnosis_counts = df['diagnostic'].value_counts()
print("5) Most common diagnoses:")
for diag, count in diagnosis_counts.items():
    print(f"{diag}: {count}")


# 2) Dataset Visualisation

# a. Cancer % from biopsies (pie chart)
if 'biopsed' in df.columns:
    biopsy_counts = df['biopsed'].value_counts()
    labels = ['Not Biopsied', 'Biopsied'] if set(biopsy_counts.index) == {0, 1} else biopsy_counts.index
    plt.figure()
    biopsy_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, labels=labels)
    plt.title('Percentage of Patients Biopsied')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

# b. Histograms for all numeric columns (excluding lesion_id)
numeric_cols = df.select_dtypes(include=['number']).columns.drop('lesion_id', errors='ignore')
for col in numeric_cols:
    plt.figure()
    df[col].dropna().hist()
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# 3. Further Exploration
# Contingency Tables + Chi squared Tests for Categorial Variables with Binary data.
# a)  done for smoke, drink, skin_cancerhistory compared to has_cancer but can be done with all binary columns.

cancer_keywords = ['BCC', 'MEL', 'SCC', 'ACK', 'MIS'] # Defining a column which says if patient has cancer or not based on their cancer-related diagnoses
df['has_cancer'] = df['diagnostic'].apply(
    lambda x: any(keyword in str(x).upper() for keyword in cancer_keywords)
).astype(int)

def analyze_binary_relation(feature):
    if feature in df.columns:
        contingency = pd.crosstab(df[feature], df['has_cancer'])
        print(f"\nContingency table for {feature} vs has_cancer:")
        print(contingency)
        chi2, p, _, _ = chi2_contingency(contingency)
        print(f"Chi-squared test result: χ² = {chi2:.2f}, p = {p:.4f}")
    else:
        print(f"\nColumn '{feature}' not found in the DataFrame.")

# Analyze each binary factor
for col in ['smoke', 'drink', 'skin_cancer_history']: # loops for these columns - can be extended to other variables. 
    analyze_binary_relation(col)
