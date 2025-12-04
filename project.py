import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import csv
source_df = pd.read_csv("result.csv", sep=";")

# remove lines with empty values
source_df = source_df.dropna()

# check if there are duplicated lines
nb_duplicates = source_df.duplicated().sum()

# and remove them if there are
if(nb_duplicates > 0):
    source_df = source_df.drop_duplicates()
    print(nb_duplicates, "duplicates removed")

# we can remove the id field as it has no impact on analytics
source_df = source_df.drop('Id', axis = 1)

# change column types
# source_df['Id'] = source_df['Id'].astype(np.int32)
source_df['gaming_interest_score'] = source_df['gaming_interest_score'].astype(np.int16)
source_df['insta_design_interest_score'] = source_df['insta_design_interest_score'].astype(np.int16)
source_df['football_interest_score'] = source_df['football_interest_score'].astype(np.int16)
source_df['campaign_success'] = source_df['campaign_success'].astype(bool)
source_df['age'] = source_df['age'].astype(np.int8)

print("dimensions :", source_df.shape)
print("types :", source_df.dtypes)


score_cols = [
    "gaming_interest_score",
    "insta_design_interest_score",
    "football_interest_score",
    "age"
]

threshold = 3

# detext anomalies by columns
for col in score_cols:
    z = (source_df[col] - source_df[col].mean()) / source_df[col].std()
    source_df[f"{col}_anomaly"] = np.abs(z) > threshold

    plt.figure(figsize=(8,5))
    plt.title(f"Anomalies — {col}")

    mean = source_df[col].mean()
    std = source_df[col].std()

    plt.axhline(mean, linestyle="--", color="black")
    plt.axhline(mean + std*threshold, linestyle="--", color="red")
    plt.axhline(mean - std*threshold, linestyle="--", color="red")

    plt.scatter(source_df.index[~source_df[f"{col}_anomaly"]], source_df[col][~source_df[f"{col}_anomaly"]], s=10, color="blue")
    plt.scatter(source_df.index[source_df[f"{col}_anomaly"]], source_df[col][source_df[f"{col}_anomaly"]], s=20, color="red")

    plt.tight_layout()
    plt.savefig(f"anomaly_{col}.png")
    plt.close()