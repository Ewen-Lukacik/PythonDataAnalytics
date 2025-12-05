import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################################# PHASE 1
# import csv
source_df = pd.read_csv("result.csv", sep=";")

# remove lines with empty values
source_df = source_df.dropna()

# check if there are duplicated lines
nb_duplicates = source_df.duplicated().sum()

# and remove them if there are
if(nb_duplicates > 0):
    source_df = source_df.drop_duplicates()
    # print(nb_duplicates, "duplicates removed")

# we can remove the id field as it has no impact on analytics
source_df = source_df.drop('Id', axis = 1)

# change column types
# source_df['Id'] = source_df['Id'].astype(np.int32)
source_df['gaming_interest_score'] = source_df['gaming_interest_score'].astype(np.int16)
source_df['insta_design_interest_score'] = source_df['insta_design_interest_score'].astype(np.int16)
source_df['football_interest_score'] = source_df['football_interest_score'].astype(np.int16)
source_df['campaign_success'] = source_df['campaign_success'].apply(lambda x: str(x).strip() in ["1", "True", "true"])
source_df['age'] = source_df['age'].astype(np.int8)

# print("dimensions :", source_df.shape)
# print("types :", source_df.dtypes)


############################################## PHASE 2
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


############################################################### PHASE 3

# cleaning up bad values
# putting everythign to lowercase (prevent differenciation like Facebook != facebook)
source_df["canal_recommande"] = source_df["canal_recommande"].str.lower().str.strip()

# getting rid of "undefined" values
source_df = source_df[source_df["canal_recommande"] != "non_defini"]

# keep obly 18+ age
source_df = source_df[source_df["age"] >= 18]

# replace mispeeled values
if "recommended_product" in source_df.columns:
    source_df["recommended_product"] = source_df["recommended_product"].replace({
        "Fornite": "Fortnite",
        "fornite": "Fortnite"
    })

    # getting rid of "Test" values
    source_df = source_df[source_df["recommended_product"] != "Test"]



# kpi 1 : global success
global_success = source_df["campaign_success"].mean()
print("global success :", round(global_success, 3))

# kpi 2 : success by reco product
kpi_product = source_df.groupby("recommended_product")["campaign_success"].mean().sort_values(ascending=False)
print("success by product :", kpi_product)

# renderthe graphic
plt.figure(figsize=(10,4))
kpi_product.plot(kind="bar")
plt.title("success rate by product")
plt.ylabel("success rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("kpi_success_by_product.png")
plt.close()


# kpi 3 : success by canal
kpi_canal = source_df.groupby("canal_recommande")["campaign_success"].mean().sort_values(ascending=False)
print("success by canal :", kpi_canal)

# render the graphic
plt.figure(figsize=(10,4))
kpi_canal.plot(kind="bar")
plt.title("success rate by canal")
plt.ylabel("success rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("kpi_success_by_canal.png")
plt.close()


# kpi 4 : success by interests (higher or lower than median)
score_cols_kpi = [
    "gaming_interest_score",
    "insta_design_interest_score",
    "football_interest_score"
]

for col in score_cols_kpi:
    median_val = source_df[col].median()
    source_df[col + "_high"] = source_df[col] > median_val

    kpi_interest = source_df.groupby(col + "_high")["campaign_success"].mean()

    print(f"Success rate by {col} (high vs low, median={median_val}) :")
    print(kpi_interest)

    # render a graphic fore ach of the interests
    plt.figure(figsize=(6,4))
    kpi_interest.plot(kind="bar")
    plt.title(f"syuccess rate — {col} (high vs low)")
    plt.ylabel("success rate")
    plt.tight_layout()
    plt.savefig(f"kpi_success_{col}.png")
    plt.close()


# kpi 5 : success by age
# we don't wznt to take into account people under 18
bins = [18, 25, 35, 50, 100]
labels = ["19-25", "26-35", "36-50", "50+"]

source_df["age_group"] = pd.cut(source_df["age"], bins=bins, labels=labels, right=False)

kpi_age = source_df.groupby("age_group")["campaign_success"].mean()
print("success rate by age :", kpi_age)

# render the graphic
plt.figure(figsize=(8,4))
kpi_age.plot(kind="bar")
plt.title("success rate by age")
plt.ylabel("success rate")
plt.tight_layout()
plt.savefig("kpi_success_by_age.png")
plt.close()

