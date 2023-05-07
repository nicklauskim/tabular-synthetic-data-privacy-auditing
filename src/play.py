import json
import pandas as pd
import numpy as np


texas = pd.read_csv("data/adult.csv", index_col=None)
print(texas.head())

for i in range(len(texas.columns)):
    print(texas.columns[i], "\n", sorted(texas.iloc[:, i].unique()), "\n")

for i in range(12):
    print(texas.columns[i], "\n", texas.iloc[:, i].unique(), "\n")

for i in range(12, 18):
    print(texas.columns[i], max(texas.iloc[:, i]) - min(texas.iloc[:, i]))