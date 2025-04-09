# TO-DO: -fix confusion matrix, so it saves to png, shows exact percentage, and only of category.
#        -calculate if majority conflict occur mostly on 0vs1 or 0vs2 or 1vs2
#        -conclusion maybe

from statsmodels.stats.inter_rater import fleiss_kappa
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


# reads csvs seperated by , and by ;
def read_csv_flexible(filepath, min_columns = 2): 
        df =  pd.read_csv(filepath,delimiter=";")
        if df.shape[1] >= min_columns:
            return df
        return pd.read_csv(filepath,delimiter=",")
        
def convert_to_ratings(df):
    # Drop rows with any missing ratings 
    df_clean = df.dropna()
    ratings_cols = [col for col in df_clean.columns.tolist() if "Rating" in col]
    # Convert ratings to integers
    ratings = df_clean[ratings_cols].astype(int)
    return ratings

# calculates the kappa score and formats the df
def kappa_score_calc(ratings):
    # get the counts of different annotations for each picture
    counts = [[img.count(0),img.count(1),img.count(2)] for img in ratings.itertuples(index=False, name=None)]
    return fleiss_kappa(counts)

# returns the majority label, if none exists return -1
def get_majority(row):
    majority, count = mode(row)
    if count < len(row)/2:
        return -1   
    else: 
        return majority
    
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

i=1
for df in dataframes:
   
    ratings = convert_to_ratings(df)
    
    kappa_scores[df.at[2,"Group_ID"]] = kappa_score_calc(ratings)  # calculate kappa-score
    ratings["Standard_deviation"] = ratings.std(axis=1) # cacluates standard deviation, higher std means more disagreement
    ratings["Majority_label"] = ratings.apply(get_majority, axis = 1) # calculates the majority annotation
    majority_counts = ratings["Majority_label"].value_counts().sort_index()


    bars = plt.bar(majority_counts.index, majority_counts.values, color = [{-1:"gray", 0:"orange", 1:"skyblue", 2:"green"}[cat] for cat in majority_counts.index])
    plt.xticks([-1,0,1,2], ["No majority","0(None)","1(Some)","2(A lot)"])
    plt.xlabel("Hair Density Rating")
    plt.ylabel("Percentage")
    plt.title(f"Overall Distribution of Group {df.at[2,"Group_ID"]}, n={ratings.shape[0]}")
    plt.bar_label(bars, labels=[f"{p}%" for p in (majority_counts/majority_counts.sum()*100).round(1)])
    plt.tight_layout()
    plt.savefig(project_root / "result" / f"distributions{df.at[2,"Group_ID"]}.png")
    plt.clf()
    print(df.at[2,"Group_ID"])
    print(ratings)



    
    
result = pd.DataFrame.from_dict(kappa_scores, orient="index",columns=["Kappa_score"])
print(result)
#Interpretation:
#κ < 0.20: Slight
#0.21–0.40: Fair
#0.41–0.60: Moderate
#0.60: Substantial
# group J has the lowest k score= 0.586 and group B has the highest k score=0.88, lets analyze them further

print(dataframes)
# group J -> 5 annotators and n=200, need to take into consideration
for i in range(1,5):
    for j in range(i,5):
        raterA = f"Rating_{i}"
        raterB = f"Rating_{j}"
        ratings = convert_to_ratings(dataframes[1])
        print(dataframes[2])
        #Create the confusion matrix
        cm = confusion_matrix(ratings[raterA],ratings[raterB]) 
        
        #Plotting
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel(raterB + "Ratings")
        plt.ylabel(raterA + "Ratings")
        plt.xticks([0, 1, 2])
        plt.yticks([0, 1, 2])
        plt.title(raterA + "vs." + raterB + "Agreement")
        plt.show()
        plt.clf()
        
    

    