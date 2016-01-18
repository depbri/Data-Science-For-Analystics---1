###################################
####### 0 - Package
###################################

import pandas as pandas
import numpy as np
import matplotlib.pyplot as mplot
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import re
import operator


###################################
####### 1 - Data
###################################

#####
# Import
#####
Train = pandas.read_csv("C:/DEV/Kaggle/Titanic/Inputs/train.csv")
print(Train.head())
print(Train.describe())

Test = pandas.read_csv("C:/DEV/Kaggle/Titanic/Inputs/test.csv")
print(Test.head())
print(Test.describe())

Data = pandas.concat([Train,Test])
Data = Data.drop(["Survived"], axis=1) 
print(Data.head())
print(Data.describe())

#####
# Missing value
#####

# Train
print(pandas.isnull(Train["PassengerId"]).nonzero())
print(pandas.isnull(Train["Survived"]).nonzero())
print(pandas.isnull(Train["Pclass"]).nonzero())
print(pandas.isnull(Train["Name"]).nonzero())
print(pandas.isnull(Train["Sex"]).nonzero())
print(pandas.isnull(Train["Age"]).nonzero())
print(pandas.isnull(Train["SibSp"]).nonzero())
print(pandas.isnull(Train["Parch"]).nonzero())
print(pandas.isnull(Train["Ticket"]).nonzero())
print(pandas.isnull(Train["Fare"]).nonzero())
print(pandas.isnull(Train["Cabin"]).nonzero())
print(pandas.isnull(Train["Embarked"]).nonzero())

# Test
print(pandas.isnull(Test["PassengerId"]).nonzero())
print(pandas.isnull(Test["Pclass"]).nonzero())
print(pandas.isnull(Test["Name"]).nonzero())
print(pandas.isnull(Test["Sex"]).nonzero())
print(pandas.isnull(Test["Age"]).nonzero())
print(pandas.isnull(Test["SibSp"]).nonzero())
print(pandas.isnull(Test["Parch"]).nonzero())
print(pandas.isnull(Test["Ticket"]).nonzero())
print(pandas.isnull(Test["Fare"]).nonzero())
print(pandas.isnull(Test["Cabin"]).nonzero())
print(pandas.isnull(Test["Embarked"]).nonzero())

# Age
Train["Age"] = Train["Age"].fillna(round(Data["Age"].mean()))
Test["Age"] = Test["Age"].fillna(round(Data["Age"].mean()))

# Cabin
print(Train["Cabin"].unique())
print(Test["Cabin"].unique())
"""A Compléter"""

# Fare
print(pandas.isnull(Test["Fare"]).nonzero())
Test["Fare"][152] = Data.Fare[Data.Pclass == Test["Pclass"][152]].mean()

# Embarked
print(Train["Embarked"].unique())
Train["Embarked"] = Train["Embarked"].fillna('S')

#####
# Standadising Features
#####

# Name
"""A Compléter"""

# Ticket
"""A Compléter"""

# Cabin
"""A Compléter"""


###################################
####### 2 - Descriptive Statistics
###################################

# Survival by Sex (Proportion)
P_Survival_Male = Train.Survived[Train.Sex == 'male'].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'male'].size)
P_Survival_Female = Train.Survived[Train.Sex == 'female'].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'female'].size)
P_Sex = Train.Sex.value_counts().sort_index(ascending=False)/Train.Sex.size
mplot.bar(np.cumsum(np.concatenate(([0],P_Sex[0:1])))+np.array([0,0.05]),[P_Survival_Male[1] , P_Survival_Female[1]], color='grey',label='Survival',width=P_Sex)
mplot.bar(np.cumsum(np.concatenate(([0],P_Sex[0:1])))+np.array([0,0.05]),[P_Survival_Male[0] , P_Survival_Female[0]], bottom = [P_Survival_Male[1] , P_Survival_Female[1]], color='grey', alpha = 0.5, label='Death',width=P_Sex)
mplot.xlim(-0.1,1+0.05*2+0.1)
mplot.ylim(0,1.1)
mplot.xticks(np.cumsum(np.concatenate(([0],P_Sex[0:1])))+P_Sex/2+np.array([0,0.05]),["Male","Female"])
mplot.legend()
mplot.show()

# Survival by Class (Proportion)
P_Survival_Class_1 = Train.Survived[Train.Pclass == 1].value_counts().sort_index()/float(Train.Pclass[Train.Pclass == 1].size)
P_Survival_Class_2 = Train.Survived[Train.Pclass == 2].value_counts().sort_index()/float(Train.Pclass[Train.Pclass == 2].size)
P_Survival_Class_3 = Train.Survived[Train.Pclass == 3].value_counts().sort_index()/float(Train.Pclass[Train.Pclass == 3].size)
P_Class = Train.Pclass.value_counts().sort_index()/Train.Pclass.size
mplot.bar(np.cumsum(np.concatenate(([0],P_Class[0:2])))+np.array([0,0.05,0.1]),[P_Survival_Class_1[1] , P_Survival_Class_2[1] , P_Survival_Class_3[1]], color='grey',label='Survival',width=P_Class)
mplot.bar(np.cumsum(np.concatenate(([0],P_Class[0:2])))+np.array([0,0.05,0.1]),[P_Survival_Class_1[0] , P_Survival_Class_2[0] , P_Survival_Class_3[0]], bottom = [P_Survival_Class_1[1] , P_Survival_Class_2[1] , P_Survival_Class_3[1]], color='grey', alpha = 0.5, label='Death',width=P_Class)
mplot.xlim(-0.1,1+0.05*2+0.1)
mplot.ylim(0,1.1)
mplot.xticks(np.cumsum(np.concatenate(([0],P_Class[0:2])))+P_Class/2+np.array([0,0.05,0.1]),["Class 1","Class 2","Class 3"])
mplot.legend()
mplot.show()

# Survival by Embarked (Proportion)
P_Survival_Embarked_C = Train.Survived[Train.Embarked == 'C'].value_counts().sort_index()/float(Train.Embarked[Train.Embarked == 'C'].size)
P_Survival_Embarked_Q = Train.Survived[Train.Embarked == 'Q'].value_counts().sort_index()/float(Train.Embarked[Train.Embarked == 'Q'].size)
P_Survival_Embarked_S = Train.Survived[Train.Embarked == 'S'].value_counts().sort_index()/float(Train.Embarked[Train.Embarked == 'S'].size)
P_Embarked = Train.Embarked.value_counts().sort_index()/Train.Embarked.size
mplot.bar(np.cumsum(np.concatenate(([0],P_Embarked[0:2])))+np.array([0,0.05,0.1]),[P_Survival_Embarked_C[1] , P_Survival_Embarked_Q[1] , P_Survival_Embarked_S[1]], color='grey',label='Survival',width=P_Embarked)
mplot.bar(np.cumsum(np.concatenate(([0],P_Embarked[0:2])))+np.array([0,0.05,0.1]),[P_Survival_Embarked_C[0] , P_Survival_Embarked_Q[0] , P_Survival_Embarked_S[0]], bottom = [P_Survival_Embarked_C[1] , P_Survival_Embarked_Q[1] , P_Survival_Embarked_S[1]], color='grey', alpha = 0.5, label='Death',width=P_Embarked)
mplot.xlim(-0.1,1+0.05*2+0.1)
mplot.ylim(0,1.1)
mplot.xticks(np.cumsum(np.concatenate(([0],P_Embarked[0:2])))+P_Embarked/2+np.array([0,0.05,0.1]),["C","Q","S"])
mplot.legend()
mplot.show()

# Survival by Gender and Class (Proportion)
fig = mplot.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
P_Female_Survival_High_Class = Train.Survived[Train.Sex == 'female'][Train.Pclass == 1].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'female'][Train.Pclass == 1].size)
P_Female_Survival_Middle_Class = Train.Survived[Train.Sex == 'female'][Train.Pclass == 2].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'female'][Train.Pclass == 2].size)
P_Female_Survival_Low_Class = Train.Survived[Train.Sex == 'female'][Train.Pclass == 3].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'female'][Train.Pclass == 3].size)
ax1.bar(np.array(range(3)),[P_Female_Survival_High_Class[1] , P_Female_Survival_Middle_Class[1] , P_Female_Survival_Low_Class[1]], color='pink',label='Survival',width=0.3)
ax1.bar(np.array(range(3)),[P_Female_Survival_High_Class[0] , P_Female_Survival_Middle_Class[0] , P_Female_Survival_Low_Class[0]], bottom = [P_Female_Survival_High_Class[1] , P_Female_Survival_Middle_Class[1] , P_Female_Survival_Low_Class[1]], color='pink', alpha = 0.5, label='Death',width=0.3)
ax1.set_xlim(-0.3,2+0.3*2)
ax1.set_ylim(0,1.1)
ax1.set_xticks(np.array(range(3))+0.15)
ax1.set_xticklabels(["Female, High","Female, Middle","Female, Low"])
ax1.legend()
ax2 = fig.add_subplot(122)
P_Male_Survival_High_Class = Train.Survived[Train.Sex == 'male'][Train.Pclass == 1].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'male'][Train.Pclass == 1].size)
P_Male_Survival_Middle_Class = Train.Survived[Train.Sex == 'male'][Train.Pclass == 2].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'male'][Train.Pclass == 2].size)
P_Male_Survival_Low_Class = Train.Survived[Train.Sex == 'male'][Train.Pclass == 3].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'male'][Train.Pclass == 3].size)
ax2.bar(np.array(range(3)),[P_Male_Survival_High_Class[1] , P_Male_Survival_Middle_Class[1] , P_Male_Survival_Low_Class[1]], color='blue',label='Survival',width=0.3)
ax2.bar(np.array(range(3)),[P_Male_Survival_High_Class[0] , P_Male_Survival_Middle_Class[0] , P_Male_Survival_Low_Class[0]], bottom = [P_Male_Survival_High_Class[1] , P_Male_Survival_Middle_Class[1] , P_Male_Survival_Low_Class[1]], color='blue', alpha = 0.5, label='Death',width=0.3)
ax2.set_xlim(-0.3,2+0.3*2)
ax2.set_ylim(0,1.1)
ax2.set_xticks(np.array(range(3))+0.15)
ax2.set_xticklabels(["Male, High","Male, Middle","Male, Low"])
ax2.legend()
mplot.show()

# Age Distribution by classes
Train.Age[Train.Pclass == 1].plot(kind='kde')    
Train.Age[Train.Pclass == 2].plot(kind='kde')
Train.Age[Train.Pclass == 3].plot(kind='kde')
mplot.xlabel("Age")    
mplot.title("Age Distribution within classes")
mplot.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 
mplot.show()


###################################
####### 3, 4 & 5 - Train Models, Test Models & Model Selection
###################################

#####
# Standardisation
#####

# From string to integer
Train.loc[Train["Sex"] == "male", "Sex"] = 0
Train.loc[Train["Sex"] == "female", "Sex"] = 1

Train.loc[Train["Embarked"] == "S", "Embarked"] = 0
Train.loc[Train["Embarked"] == "C", "Embarked"] = 1
Train.loc[Train["Embarked"] == "Q", "Embarked"] = 2

#####
# Generating New Features
#####

# Family size
Train["FamilySize"] = Train["SibSp"] + Train["Parch"]

# Family Group
family_id_mapping = {} # A dictionary mapping family name to id

def get_family_id(row): # A function to get the id given a row
    last_name = row["Name"].split(",")[0] # Find the last name by splitting on a comma
    family_id = "{0}{1}".format(last_name, row["FamilySize"]) # Create the family id
    if family_id not in family_id_mapping: # Look up the id in the mapping
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

family_ids = Train.apply(get_family_id, axis=1) # Get the family ids with the apply method
family_ids[Train["FamilySize"] < 3] = -1 # There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
print(pandas.value_counts(family_ids)) # Print the count of each unique id.
Train["FamilyId"] = family_ids

# Name length
Train["NameLength"] = Train["Name"].apply(lambda x: len(x))

# Titles
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search: # If the title exists, extract and return it.
        return title_search.group(1)
    return ""

titles = Data["Name"].apply(get_title) # Get all the titles and print how often each one occurs.
print(pandas.value_counts(titles))
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Dona": 10, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

titles = Train["Name"].apply(get_title)
print(pandas.value_counts(titles))
for k,v in title_mapping.items():
    titles[titles == k] = v
print(pandas.value_counts(titles))
Train["Title"] = titles

#####
# Linear Regression
#####

Train_Light = Train.drop(['Ticket','Cabin','Name','FamilySize','NameLength'], axis=1)

# Split into 3 subsbset
kf = KFold(Train_Light.shape[0], n_folds=3, random_state=1)

# Regreassion
alg = LinearRegression()
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
predictions = []

for train, test in kf:
    train_predictors = (Train_Light[predictors].iloc[train,:])
    train_target = Train_Light["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(Train_Light[predictors].iloc[test,:])
    predictions.append(test_predictions)

# Evaluating Error
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
accuracy = sum(Train_Light["Survived"] == predictions)/float(Train_Light["Survived"].count())

#####
# Logistic Regression
#####

alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, Train_Light[predictors], Train_Light["Survived"], cv=3)
print(scores.mean())

#####
# Random Forest
#####

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
scores = cross_validation.cross_val_score(alg, Train[predictors], Train["Survived"], cv=3)
print(scores.mean())

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores = cross_validation.cross_val_score(alg, Train[predictors], Train["Survived"], cv=3)
print(scores.mean())

# Features Selection
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(Train[predictors], Train["Survived"])
scores = -np.log10(selector.pvalues_) # Get the raw p-values for each feature, and transform from p-values into scores

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
mplot.bar(range(len(predictors)), scores)
mplot.xticks(range(len(predictors)), predictors, rotation='vertical')
mplot.show()

# Pick only the four best features.
predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
scores = cross_validation.cross_val_score(alg, Train[predictors], Train["Survived"], cv=3)
print(scores.mean())

#####
# Gradient boosting
#####

# The algorithms we want to ensemble.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

kf = KFold(Train.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    train_target = Train["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(Train[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(Train[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == Train["Survived"]]) / len(predictions)
print(accuracy)


###################################
####### 6 - Prediction Submission
###################################


# From string to integer
Test.loc[Test["Sex"] == "male", "Sex"] = 0
Test.loc[Test["Sex"] == "female", "Sex"] = 1

Test.loc[Test["Embarked"] == "S", "Embarked"] = 0
Test.loc[Test["Embarked"] == "C", "Embarked"] = 1
Test.loc[Test["Embarked"] == "Q", "Embarked"] = 2

# Family size
Test["FamilySize"] = Test["SibSp"] + Test["Parch"]

# Family Group
family_ids = Test.apply(get_family_id, axis=1) # Get the family ids with the apply method
family_ids[Test["FamilySize"] < 3] = -1 # There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
Test["FamilyId"] = family_ids

# Name length
Test["NameLength"] = Test["Name"].apply(lambda x: len(x))

# Titles
titles = Test["Name"].apply(get_title)
print(pandas.value_counts(titles))
for k,v in title_mapping.items():
    titles[titles == k] = v
print(pandas.value_counts(titles))
Test["Title"] = titles

#####
# Logistic Regression
#####

Test_Light = Test.drop(['Ticket','Cabin','Name','FamilySize','NameLength'], axis=1)

# Train the algorithm using all the training data
alg = LogisticRegression(random_state=1)
alg.fit(Train_Light[predictors], Train_Light["Survived"])
predictions = alg.predict(Test_Light[predictors])
submission = pandas.DataFrame({
        "PassengerId": Test_Light["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("C:/DEV/Kaggle/Titanic/Outputs/submission_LogisticRegression.csv",index = False)  

#####
# Random Forest
#####
"""A Compléter"""

#####
# Gradient boosting
#####

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
full_predictions = []

for alg, predictors in algorithms:
    alg.fit(Train[predictors], Train["Survived"])
    predictions = alg.predict_proba(Test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)
 
# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

# Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission = pandas.DataFrame({
        "PassengerId": Test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("C:/DEV/Kaggle/Titanic/Outputs/submission_GradientBoosting.csv",index = False)  
