#####
# Packages
#####

import pandas as pandas
import matplotlib.pyplot as mplot

#####
# Data
#####

# Import 
Train = pandas.read_csv("C:/DEV/Kaggle/Titanic/Inputs/train.csv")
print(Train.head())
print(Train.describe())

# Exploration
Train.Survived.value_counts().plot(kind='bar')
mplot.title("Distribution of Survival, (1 = Survived)")  

"""Reprendre ICI"""

#####
# Missing value
#####

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

# Age
Train["Age"] = Train["Age"].fillna(round(Train["Age"].mean()))

# Cabin
"""Reprendre ICI"""

# Embarked
