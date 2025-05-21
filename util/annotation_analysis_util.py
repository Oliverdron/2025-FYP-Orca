# TO-DO: 
#        -calculate if majority conflict occur mostly on 0vs1 or 0vs2 or 1vs2
#        -conclusion maybe

from statsmodels.stats.inter_rater import fleiss_kappa
from scipy.stats import mode
import seaborn as sns
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
dataframes = {}
for path in annotation_path.iterdir():
    if str(path).endswith(".csv"):
        dataframe = read_csv_flexible(path)
        #dataframe.info()
        unique_group_id = dataframe['Group_ID'].unique()[0]
        dataframes[f"Group_{unique_group_id}"] = dataframe
        
        
# there are 4 or 5 annotators
ratings_cols = ["Rating_1", "Rating_2", "Rating_3", "Rating_4", "Rating_5"] 

kappa_scores = {} # save the kappa scores 

i=1
for group, df in dataframes.items():
    ratings_cols = [col for col in df.columns if col.startswith("Rating_")]
    ratings = convert_to_ratings(df)
    
    kappa_scores[group] = kappa_score_calc(ratings)  # calculate kappa-score
    ratings["Standard_deviation"] = round(ratings.std(axis=1),3) # cacluates standard deviation, higher std means more disagreement
    ratings["Majority_label"] = ratings.apply(get_majority, axis = 1) # calculates the majority annotation
    majority_counts = ratings["Majority_label"].value_counts().sort_index()
        
    no_majority = ratings[ratings["Majority_label"] == -1]

    # Create a list to store the types of disagreements
    conflict_pairs = []

    # Go through each row of ratings (for images without a clear majority)
    for row in no_majority[ratings_cols].itertuples(index=False):
        # Convert the ratings to a set to find unique values (e.g., {0, 1})
        unique_values = set(row)

        # If exactly 2 unique ratings exist, it's a clear pair like (0,1)
        if len(unique_values) == 2:
            pair = tuple(sorted(unique_values))  # Sort for consistent order
            conflict_pairs.append(pair)

        # If all 3 ratings are present (0, 1, and 2), mark as "mixed"
        elif len(unique_values) == 3:
            conflict_pairs.append(("mixed",))

    # Convert the list of conflicts into a pandas Series to count frequencies
    pair_counts = pd.Series(conflict_pairs).value_counts()

    # Plot the counts as a bar chart
    if not pair_counts.empty:
        pair_counts.plot(kind="bar", title=f"Conflicting Pairs in No-Majority Cases for {group}")
        plt.xlabel(f"Conflict Type (Rating Pairs)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(project_root / "result" / "annotation_analysis" / "Majority_conflict_types" /  f"distributions{df.at[2,"Group_ID"]}.png")
    else:
        print(f"no majorities in {group}")
    plt.clf()


    melted_ratings = pd.melt(ratings[ratings_cols], value_vars=ratings_cols, 
                            var_name="Rater", value_name="Rating")

    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x="Rating", hue="Rater", data=melted_ratings, palette="Set2")

    # Customize plot
    plt.xlabel("Rating Value")
    plt.ylabel("Frequency")
    plt.title(f"Rating Distribution per Annotator for {group}")
    plt.xticks([0, 1, 2], ["0 (None)", "1 (Some)", "2 (A lot)"], rotation=0)
    plt.legend(title="Raters", bbox_to_anchor=(1.05, 1), loc='upper left')
    # Add mean and std annotations for each rater on the side
    y_offset = 0
    for col in ratings_cols:
        mean_rating = ratings[col].mean()
        std_rating = ratings[col].std()
        text = f"{col}: Mean = {mean_rating:.2f}, Std = {std_rating:.2f}"
        plt.text(2.5, y_offset, text, ha='left', va='bottom', fontsize=10, color='black')
        y_offset -= 2  # Adjust vertical space between each rater's text
    plt.tight_layout()
    plt.savefig(project_root / "result" / "annotation_analysis" / "Distributions" /  f"Distribution_per_annotator{group}.png")
    plt.clf()




    bars = plt.bar(majority_counts.index, majority_counts.values, color = [{-1:"gray", 0:"orange", 1:"skyblue", 2:"green"}[cat] for cat in majority_counts.index])
    plt.xticks([-1,0,1,2], ["No majority","0(None)","1(Some)","2(A lot)"])
    plt.xlabel("Hair Density Rating")
    plt.ylabel("Percentage")
    plt.title(f"Overall Distribution of Group {df.at[2,"Group_ID"]}, n={ratings.shape[0]}")
    plt.bar_label(bars, labels=[f"{p}%" for p in (majority_counts/majority_counts.sum()*100).round(1)])
    plt.tight_layout()
    plt.savefig(project_root / "result" / "annotation_analysis" / "Distributions" /  f"distributions_{group}.png")
    plt.clf()
    #
    #print(df.at[2,"Group_ID"])
    #print(ratings)
    # Define where to save it
    output_path = project_root / "result" / "annotation_analysis" / "Ratings_analysis" /f"ratings_{group}_cleaned.csv"

    # Save the DataFrame
    ratings.to_csv(output_path, index=False)



    
    
result = pd.DataFrame.from_dict(kappa_scores, orient="index",columns=["Kappa_score"])
#print(result)
#   Kappa_score
#O     0.708008
#G     0.699370
#J     0.586344
#B     0.880848
#N     0.709016
#Interpretation:
#κ < 0.20: Slight
#0.21–0.40: Fair
#0.41–0.60: Moderate
#0.60: Substantial
# group J has the lowest k score= 0.586 and group B has the highest k score=0.88, lets analyze them further

#print(dataframes)
# group J -> 5 annotators and n=200, need to take into consideration
for i in range(1,6):
    for j in range(i+1,6):
        group = "Group J"
        raterA = f"Rating_{i}"
        raterB = f"Rating_{j}"
        ratings = convert_to_ratings(dataframes["Group_J"])
        ratingsA = ratings[raterA]
        ratingsB = ratings[raterB]
        labels = [0,1,2]
        cm = confusion_matrix(ratingsA,ratingsB)
        cm_percent = cm / cm.sum()
        
        correct = np.trace(cm)  # Sum of diagonal elements
        total = cm.sum()
        # Flatten ratings
        differences = np.abs(ratingsA - ratingsB)

        # Calculate counts
        total_pairs = len(differences)
        agree = np.sum(differences == 0)
        weak_disagree = np.sum(differences == 1)
        strong_disagree = np.sum(differences == 2)

        # Convert to percentages
        agree_pct = agree / total_pairs
        weak_disagree_pct = weak_disagree / total_pairs
        strong_disagree_pct = strong_disagree / total_pairs
        
        
        
        ax = sns.heatmap(cm_percent, annot=True, 
            fmt='.0%', cmap='Blues', xticklabels=labels, yticklabels=labels)
        ax.set_xlabel(f"Rater {i}")
        ax.set_ylabel(f"Rater {j}")
        ax.set_title(f"Rater {i} vs Rater {j} in {group}")
        annotation_text = (
                f"Agreement: {agree_pct:.1%}\n"
                f"Mild disagreement (Δ=1): {weak_disagree_pct:.1%}\n"
                f"Strong disagreement (Δ=2): {strong_disagree_pct:.1%}\n"
                f"N = 200 (images)")
        plt.text(0, -0.4, annotation_text, fontsize=10, ha='left', va='center', transform=ax.transAxes)

        output_path = project_root / "result" / "annotation_analysis" / "heatmaps" / group / f"{raterA}_{raterB}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
# Group B -> only 4 annotators n = 99
for i in range(1,5):
    for j in range(i+1,5):
        group = "Group B"
        raterA = f"Rating_{i}"
        raterB = f"Rating_{j}"
        ratings = convert_to_ratings(dataframes["Group_B"])
        ratingsA = ratings[raterA]
        ratingsB = ratings[raterB]
        labels = [0,1,2]
        cm = confusion_matrix(ratingsA,ratingsB)
        cm_percent = cm / cm.sum()
        
        correct = np.trace(cm)  # Sum of diagonal elements
        total = cm.sum()
        accuracy = correct / total
        disagreement = 1 - accuracy
        
        
        # Flatten ratings
        differences = np.abs(ratingsA - ratingsB)

        # Calculate counts
        total_pairs = len(differences)
        agree = np.sum(differences == 0)
        weak_disagree = np.sum(differences == 1)
        strong_disagree = np.sum(differences == 2)

        # Convert to percentages
        agree_pct = agree / total_pairs
        weak_disagree_pct = weak_disagree / total_pairs
        strong_disagree_pct = strong_disagree / total_pairs
        
        
        ax = sns.heatmap(cm_percent, annot=True, 
            fmt='.0%', cmap='Blues', xticklabels=labels, yticklabels=labels)
        ax.set_xlabel(f"Rater {i}")
        ax.set_ylabel(f"Rater {j}")
        ax.set_title(f"Rater {i} vs Rater {j} in {group}")
        annotation_text = (
                f"Agreement: {agree_pct:.1%}\n"
                f"Weak disagreement (Δ=1): {weak_disagree_pct:.1%}\n"
                f"Strong disagreement (Δ=2): {strong_disagree_pct:.1%}\n"
                f"N = 100 (images)")
        plt.text(0, -0.4, annotation_text, fontsize=10, ha='left', va='center', transform=ax.transAxes)

        output_path = project_root / "result" / "annotation_analysis" /  "heatmaps" / group / f"{raterA}_{raterB}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        



