###################################
####### 0 - Package
###################################

import pandas as pandas
import numpy as np
import matplotlib.pyplot as mplot
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import re
import operator


###################################
####### 1 - Data
###################################

#####
# Import
#####
Train = pandas.read_csv("C:/DEV/Kaggle/Titanic/Inputs/train.csv")
Test = pandas.read_csv("C:/DEV/Kaggle/Titanic/Inputs/test.csv")
Data = pandas.concat([Train,Test])
Data = Data.drop(["Survived"], axis=1) 
#print(Data.head())
#print(Data.describe())

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

# From string to integer
Train.loc[Train["Sex"] == "male", "Sex"] = 0
Train.loc[Train["Sex"] == "female", "Sex"] = 1
Test.loc[Test["Sex"] == "male", "Sex"] = 0
Test.loc[Test["Sex"] == "female", "Sex"] = 1

Train.loc[Train["Embarked"] == "S", "Embarked"] = 0
Train.loc[Train["Embarked"] == "C", "Embarked"] = 1
Train.loc[Train["Embarked"] == "Q", "Embarked"] = 2
Test.loc[Test["Embarked"] == "S", "Embarked"] = 0
Test.loc[Test["Embarked"] == "C", "Embarked"] = 1
Test.loc[Test["Embarked"] == "Q", "Embarked"] = 2


#####
# Generating New Features
#####

# Family size
Train["FamilySize"] = Train["SibSp"] + Train["Parch"] + 1
Test["FamilySize"] = Test["SibSp"] + Test["Parch"] + 1
Data["FamilySize"] = Data["SibSp"] + Data["Parch"] + 1

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

family_ids = Data.apply(get_family_id, axis=1) # Get the family ids with the apply method
family_ids[Data["FamilySize"] <= 3] = -1 # There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
print(pandas.value_counts(family_ids)) # Print the count of each unique id.

for i in range(1308):
    if int(family_ids[i:(i+1)]) in (pandas.value_counts(family_ids)[pandas.value_counts(family_ids) <= 3]).index:
        family_ids[i:(i+1)] = -1

Train["FamilyId"] = family_ids[:891]
Test["FamilyId"] = family_ids[891:]

# Titles
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search: # If the title exists, extract and return it.
        return title_search.group(1)
    return ""

titles = Data["Name"].apply(get_title) # Get all the titles and print how often each one occurs.
print(pandas.value_counts(titles))
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 9, "Mlle": 8, "Mme": 8, "Don": 7, "Dona": 10, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 7, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
print(pandas.value_counts(titles))
Train["Title"] = titles[:891]
Test["Title"] = titles[891:]

# Fare
Train.Fare[Train.Pclass == 3][ ( ( Train.Fare[Train.Pclass == 3] ) == 0 ) ] = Train.Fare[Train.Pclass == 3].median() 
Train.Fare[Train.Pclass == 2][ ( ( Train.Fare[Train.Pclass == 2] ) == 0 ) ] = Train.Fare[Train.Pclass == 2].median() 
Train.Fare[Train.Pclass == 1][ ( ( Train.Fare[Train.Pclass == 1] ) == 0 ) ] = Train.Fare[Train.Pclass == 1].median() 

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
# Random Forest
#####

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
#predictors = ["Pclass", "Sex", "Fare", "Title"]
predictors = ["Pclass", "Sex", "Fare", "Embarked", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=10, min_samples_leaf=4)
scores = cross_validation.cross_val_score(alg, Train[predictors], Train["Survived"], cv=3)
print(scores.mean())


###################################
####### 6 - Prediction Submission
###################################

#####
# Random Forest
#####

# Train the algorithm using all the training data
alg = RandomForestClassifier(random_state=1, n_estimators=1000, min_samples_split=10, min_samples_leaf=4)
alg.fit(Train[predictors], Train["Survived"])
predictions = alg.predict(Test[predictors])
submission = pandas.DataFrame({
        "PassengerId": Test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("C:/DEV/Kaggle/Titanic/Outputs/submission_RandomForestClassifier.csv",index = False) 
