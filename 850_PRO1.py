import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb
import sklearn as sk

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score

def data_processing(df):
    # Extract the target variable (train_y) and features (df_X)
    y = df["Step"]
    X = df.drop(columns=["Step"])
    
    # Scaling the features
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_data = scaler.transform(X)
    scaled_data_df = pd.DataFrame(scaled_data, columns=X.columns)
    print("processed data")
    
    return scaled_data_df, y



#for performance analysis
def getScores(true,pred):
    print("Precision score: ", precision_score(true, pred, average= 'micro'))
    print("Accuracy score: ", accuracy_score(true, pred))
    print("F1 score: ",f1_score(true, pred, average= 'micro'))
    
    
    return None

    
#STEP 1: importing data into Dataframe 
df = pd.read_csv("Project 1 Data.csv")

#STEP 2: displays data distrubution 
sns.countplot(df, x = "Step")
plt.show()

#STEP 3: corr matrix of training data only to see corrolation betwn points
corr_matrix = (df.drop(columns=["Step"])).corr()
sns.heatmap(np.abs(corr_matrix))

#STEP 4: 20, 80 split for testing, and training data.
#prep for shuffle data
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=501)
#splitting to 20% and 80% data
for train_index ,test_index in split.split(df,df["Step"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)


train_X,train_y = data_processing(strat_train_set)
test_X,test_y = data_processing(strat_test_set)


#FOR GRID SEARCH PARAMETERS AND MODELS
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# #COMMENTED OUT FOR FASTER DEBUGGING
# # model 1 random forest
# m1 = RandomForestClassifier(random_state = 501)

# params1 = {
#     'n_estimators': [10,50,100],
#     'max_depth': [None,5,10,15],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2']
# }

# print("\nrunning grid search for Random Forest Model")
# grid_search = GridSearchCV(m1, params1, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
# grid_search.fit(train_X, train_y)
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)
# best_m1 = grid_search.best_estimator_



#model 2 Support vector machine
m2 = SVC(random_state= 501)

params2 = {
    'C': [1,2,3,4,5],
    'kernel': ['linear','rbf','poly','sigmoid'],
    'gamma': ['scale','auto'],
}

print("\nrunning grid search for SVC Model")
grid_search = GridSearchCV(m2, params2, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params2 = grid_search.best_params_
print("Best Hyperparameters:", best_params2)
best_m2 = grid_search.best_estimator_

# #model 3 
# m3 = sk.tree.DecisionTreeClassifier(random_state = 501)

# params3 = {
#     'criterion': ['gini','entropy','log_loss'],
#     'splitter': ['best','random'],
#     'max_depth': [None,5,10,15],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': [1,'sqrt', 'log2']
# }
# print("\nrunning grid search for DTC Model")
# grid_search = GridSearchCV(m3, params3, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
# grid_search.fit(train_X, train_y)
# best_params3 = grid_search.best_params_
# print("Best Hyperparameters:", best_params3)
# best_m3 = grid_search.best_estimator_

#STEP 5: performance

# #model 1
# best_m1.fit(train_X,train_y)
# m1_pred = best_m1.predict(test_X)

# print("\n~~scores for random forest model~~\n")
# getScores(test_y,m1_pred)


#model 2
best_m2.fit(train_X,train_y)
m2_pred = best_m2.predict(test_X)

print("\n~~scores for SVC model~~\n")
getScores(test_y,m2_pred)
cm = confusion_matrix(test_y, m2_pred)

disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()

    





