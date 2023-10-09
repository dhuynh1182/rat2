import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn as sk

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

def data_processing(df):
    # Extract the target variable (train_y) and features (df_X)
    train_y = df["Step"]
    df_X = df.drop(columns=["Step"])
    
    # Scaling the features
    scaler = StandardScaler()
    scaler.fit(df_X)
    scaled_data = scaler.transform(df_X)
    scaled_data_df = pd.DataFrame(scaled_data, columns=df_X.columns)
    print("processed data")
    
    return scaled_data_df, train_y

    
#importing data into Dataframe
df = pd.read_csv("Project 1 Data.csv")

#prep for shuffle data
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=501)

#splitting to 20% and 80% data
for train_index ,test_index in split.split(df,df["Step"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)

train_X,train_y = data_processing(strat_train_set)

#corr matrix to see corrolation betwn points
corr_matrix = (train_X).corr()
sns.heatmap(np.abs(corr_matrix))


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
model1 = RandomForestRegressor(n_estimators=50, random_state=50)
model1.fit(train_X, train_y)
model1_predictions = model1.predict(train_X)
model1_train_mae = mean_absolute_error(model1_predictions, train_y)
print(model1_predictions)
print("Model 1 training MAE is: ", round(model1_train_mae,5))

test_X,test_y = data_processing(strat_test_set)




    





