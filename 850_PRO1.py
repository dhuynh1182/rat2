import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn as sk

from sklearn.model_selection import StratifiedShuffleSplit

#importing data into Dataframe
df = pd.read_csv("Project 1 Data.csv")

#prep for strat. data
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=501)

#strat. the data
for train_index ,test_index in split.split(df,df["Step"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)
    
