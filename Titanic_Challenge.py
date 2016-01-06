###################################
####### 0 - Package
###################################

import pandas as pandas
import numpy as np
import matplotlib.pyplot as mplot


###################################
####### 1 - Data
###################################

#####
# Import
#####
Train = pandas.read_csv("C:/DEV/Kaggle/Titanic/Inputs/train.csv")
print(Train.head())
print(Train.describe())


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
print(Train["Cabin"].unique())
"""A Compléter"""

# Embarked
print(Train["Embarked"].unique())
Train["Embarked"] = Train["Embarked"].fillna('S')


###################################
####### 2 - Descriptive Statistics
###################################

# Proportion of Survival
Train.Survived.value_counts().plot(kind='bar')
mplot.title("Distribution of Survival, (1 = Survived)")  

# Passenger Class
Train.Pclass.value_counts().plot(kind='bar')
mplot.title("Distribution of Passenger Class") 

# Sex (gender)
Train.Sex.value_counts().plot(kind='barh')
mplot.title("Distribution of Gender") 
 
# Survival by Age
mplot.scatter(Train.Survived, Train.Age)
mplot.ylabel("Age")                        
mplot.grid(b=True, which='major', axis='y')  
mplot.title("Survival by Age,  (1 = Survived)")

# Embarked (gender)
Train.Embarked.value_counts().plot(kind='bar')
mplot.title("Distribution of boarding location") 

# Age Distribution by classes
Train.Age[Train.Pclass == 1].plot(kind='kde')    
Train.Age[Train.Pclass == 2].plot(kind='kde')
Train.Age[Train.Pclass == 3].plot(kind='kde')
mplot.xlabel("Age")    
mplot.title("Age Distribution within classes")
mplot.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

# Age Distribution among Survival
Train.Age[Train.Survived == 0].plot(kind='kde')    
Train.Age[Train.Survived == 1].plot(kind='kde')
mplot.xlabel("Age")    
mplot.title("Age Distribution")
mplot.legend(('0', '1 = Survived'),loc='best') 

# Survival by Gender
Male_Survival = Train.Survived[Train.Sex == 'male'].value_counts().sort_index()
Female_Survival = Train.Survived[Train.Sex == 'female'].value_counts().sort_index()
ax = mplot.subplot(111)
ax.bar([0,1],Male_Survival, color='blue',label='Male',width=0.3)
ax.bar(np.array([0,1])+0.3,Female_Survival, color='pink', label='Female',width=0.3)
ax.set_xlim(-0.3,1+0.3*3)
ax.set_xticks(np.array([0,1])+0.3)
ax.set_xticklabels(["Death","Survival"])
ax.legend()


# Survival by Gender (Proportion)
P_Male_Survival = Train.Survived[Train.Sex == 'male'].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'male'].size)
P_Female_Survival = Train.Survived[Train.Sex == 'female'].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'female'].size)
ax = mplot.subplot(111)
ax.bar([0,1],P_Male_Survival, color='blue',label='Male',width=0.3)
ax.bar(np.array([0,1])+0.3,P_Female_Survival, color='pink', label='Female',width=0.3)
ax.set_xlim(-0.3,1+0.3*3)
ax.set_ylim(0,1)
ax.set_xticks(np.array([0,1])+0.3)
ax.set_xticklabels(["Death","Survival"])
ax.legend()
mplot.title("Who Survived? with respect to Gender and Class")

# Survival by Gender and Class
fig = mplot.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
P_Female_Survival_High_Class = Train.Survived[Train.Sex == 'female'][Train.Pclass != 3].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'female'][Train.Pclass != 3].size)
P_Female_Survival_Low_Class = Train.Survived[Train.Sex == 'female'][Train.Pclass == 3].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'female'][Train.Pclass == 3].size)
ax1.bar([0,1],P_Female_Survival_High_Class, color='pink',label='Female, High_Class',width=0.3)
ax1.bar(np.array([0,1])+0.3,P_Female_Survival_Low_Class, color='pink', alpha = 0.5, label='Female, Low_Class',width=0.3)
ax1.set_xlim(-0.3,1+0.3*3)
ax.set_ylim(0,1)
ax1.set_xticks(np.array([0,1])+0.3)
ax1.set_xticklabels(["Death","Survival"])
ax1.legend(loc='best')
ax2 = fig.add_subplot(122)
P_Male_Survival_High_Class = Train.Survived[Train.Sex == 'male'][Train.Pclass != 3].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'male'][Train.Pclass != 3].size)
P_Male_Survival_Low_Class = Train.Survived[Train.Sex == 'male'][Train.Pclass == 3].value_counts().sort_index()/float(Train.Sex[Train.Sex == 'male'][Train.Pclass == 3].size)
ax2.bar([0,1],P_Male_Survival_High_Class, color='blue',label='Male, High_Class',width=0.3)
ax2.bar(np.array([0,1])+0.3,P_Male_Survival_Low_Class, color='blue', alpha = 0.5, label='Male, Low_Class',width=0.3)
ax2.set_xlim(-0.3,1+0.3*3)
ax2.set_ylim(0,1)
ax2.set_xticks(np.array([0,1])+0.3)
ax2.set_xticklabels(["Death","Survival"])
ax2.legend(loc='best')
