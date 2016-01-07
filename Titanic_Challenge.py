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
