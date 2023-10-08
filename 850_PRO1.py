import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn as sk

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

#importing data into Dataframe
df = pd.read_csv("Project 1 Data.csv")

#prep for shuffle data
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=501)

#splitting to 20% and 80% data
for train_index ,test_index in split.split(df,df["Step"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)
    
#setting to independent variable, and dependent variable
train_y = strat_train_set['Step']
df_X = strat_train_set.drop(columns = ["Step"])
   
#scaling data 
scaler = StandardScaler()
scaler.fit(df_X)
scaled_data = scaler.transform(df_X)
scaled_data_df=pd.DataFrame(scaled_data, columns=df_X.columns)
train_X = scaled_data_df

#corr matrix to see corrolation betwn points
corr_matrix = (train_X).corr()
sns.heatmap(np.abs(corr_matrix))


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
model1 = RandomForestRegressor(n_estimators=30, random_state=50)
model1.fit(train_X, train_y)
model1_predictions = model1.predict(train_X)
model1_train_mae = mean_absolute_error(model1_predictions, train_y)
print("Model 1 training MAE is: ", round(model1_train_mae,2))



