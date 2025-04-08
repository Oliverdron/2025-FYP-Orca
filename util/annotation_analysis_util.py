from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import pandas as pd
from pathlib import Path


# reads csvs seperated by , and by ;
def read_csv_flexible(filepath, min_columns = 2): 
        df =  pd.read_csv(filepath,delimiter=";")
        if df.shape[1] >= min_columns:
            return df
        return pd.read_csv(filepath,delimiter=",")
        

# calculates the kappa score and formats the df
def kappa_score_calc(df):
    # Drop rows with any missing ratings 
    df_clean = df.dropna()
    ratings_cols = [col for col in df_clean.columns.tolist() if "Rating" in col]
    
    # Convert ratings to integers
    ratings = df_clean[ratings_cols].astype(int)
    
    # get the counts of different annotations for each picture
    counts = [[img.count(0),img.count(1),img.count(2)] for img in ratings.itertuples(index=False, name=None)]
    return fleiss_kappa(counts)
    
# Get the project root and annotation_data path
project_root = Path(__file__).parent.parent 
annotation_path = project_root / "data" / "annotation"

# Load the datasets and put them into dataframes
dataframes = []
for path in annotation_path.iterdir():
    if str(path).endswith(".csv"):
        dataframe = read_csv_flexible(path)
        #dataframe.info()
        dataframes.append(dataframe)
        
# there are 4 or 5 annotators
ratings_cols = ["Rating_1", "Rating_2", "Rating_3", "Rating_4", "Rating_5"] 

kappa_scores = {} # save the kappa scores 
std = {}

for df in dataframes:
    kappa_scores[df.at[2,"Group_ID"]] = kappa_score_calc(df)  # calculate kappa-score

    
    
result = pd.DataFrame.from_dict(kappa_scores, orient="index",columns=["Kappa_score"])
print(result)
#Interpretation:
#κ < 0.20: Slight
#0.21–0.40: Fair
#0.41–0.60: Moderate
#0.60: Substantial




    