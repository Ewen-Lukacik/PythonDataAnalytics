import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import csv
source_df = pd.read_csv("result.csv", sep=";")

# remove lines with empty values
source_df = source_df.dropna()

# change column types
source_df['Id'] = source_df['Id'].astype(np.int32)
source_df['gaming_interest_score'] = source_df['gaming_interest_score'].astype(np.int16)
source_df['insta_design_interest_score'] = source_df['insta_design_interest_score'].astype(np.int16)
source_df['football_interest_score'] = source_df['football_interest_score'].astype(np.int16)
source_df['campaign_success'] = source_df['campaign_success'].astype(bool)
source_df['age'] = source_df['age'].astype(np.int8)

# check col types
colTypes = source_df.info()
mem = source_df.memory_usage()

print(source_df.dtypes)
print(source_df.memory_usage(deep=True))