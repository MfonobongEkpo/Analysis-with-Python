#!/usr/bin/env python
# coding: utf-8

#               TITANIC DATASET ANALYSIS

#                  NOVEMBER 22, 2023

# 1 Titanic Shipreck
# 
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic
# sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone
# onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# While there was some element of luck involved in surviving, it seems some groups of people were
# more likely to survive than others. so the objective of this task is to conduct a comprehensive
# analysis on the dataset and provide a report with respect to factor to contributes to a passenger
# surviving or not
# 
# The attributes have the following meaning: Survived - that's the target, 0 means the passenger did not survive, while 1 means he/she survived. Pclass - passenger class. Name, Sex, Age - self-explanatory. SibSp - how many siblings & spouses of the passenger aboard the Titanic.

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv('tested tatanic yaf.csv')
dataset.shape


#  1.  THE DATASET

# Below is the first 10 row of the dataset showing all the columns  names we are using for the analysis which has 418,12 of the dataset information

# In[36]:


dataset.head (10)


# The Above dataset shape shows how many persons Embarked on the Titanic Shipreck.
# which is a total of 418,12.

# 2.  THE OVERALL SURVIVAL RATE

# In[3]:


overall_survival_rate = dataset['Survived'].mean() * 100


# In[4]:


print(f"the overall survival rate is: {overall_survival_rate:.2f}%")


# In[48]:


dataset.Survived.value_counts().plot(kind= 'pie', autopct = '%1.1f%%')


# The above information shows the overall survival rate of the people who Embarked on the Titanic shipreck which is 36.36%

# 3.   SURVIVAL RATE BY GENDER

# The codeind Above shows that the total number of persons who survived was only female, which means we lost all the men that Embarked on the Titanic shipreck. 100% of female survival while 0% of male survival according to the dataset. And its also represented with the bar chart below.

# In[5]:


survival_by_gender = dataset.groupby('Sex')['Survived'].mean() * 100
survival_by_gender


# In[6]:


survival_by_gender.plot(kind = 'bar', title = "Survival by gender",
ylabel = "Frequency", grid = False);


# 4.   SURVIAL RATE BY CLASS

# Below shows the number of people in a particular class.
# We have (50 presons in the 1st class),(30 presons in the 2nd class) and (72 persons in the 3rd class division).
# Out of the 50 presons in the 1st class division (46.7)persons survived
# and in the 2nd class division (32.2) persons survived while in the 3rd class division(33) persons survived.
# This information is also represented with a chart below.
# 

# In[8]:


dataset[dataset["Survived"] ==1]["Pclass"].value_counts()


# In[10]:


survival_by_class = dataset.groupby('Pclass')['Survived'].mean() * 100
survival_by_class


# In[11]:


survival_by_class.plot(kind = 'bar')
plt.title("Survival Rate by Class");


# 5.  AGE DISTRIBUTION

# The below visualization show the distributionAge of the people who Embarked on the titanic.
# this visually  shows that people in the ages of 23 has the more number of persons which is about 28 in number, followed by (20, 28 and 30) which is about (25)persopersons in each of the age of the age bracket(20,28 and 30)and people in the age bracket of(21)where just 16 in number. etc---etc.
# peole in the age of (68 and 75)had the lowers number of people that Embarked and also the oldest  of them all.
# for more explanation  please vist the visual below.

# In[12]:


dataset['Age'].hist(bins=50, grid = True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency");


# 6.  AGE DISTRIBUTION BY CLASS

# Below visual show the 3 class that we have, represented in different  color, Blue , Orange and Green in the chart below.
# the blue represent the First class which people in that class are people in the ages of 10 to 70,
# 30 has the more number of presons followed by 45 while 10 and 70 has the lowest  number of persons in that class.
# 
# Second class which is represented with the color orange has people between the ages of 0 to 50 and 60.
# people beteew the age of 20 and 30 has the highest number of people followed by 15 to 19 and 40, 50 and 60 has the same number of people in that age bracket while people in the age of 1 to 9 has the lowest  persons in the class.
# 
# The third class has the highest  number of people in that class.
# The highest age arrange in the third class are the people in the age of 19 to 25 which is about 48 people followed by 25 to 30 which are 30 in number etc.
# people in the age of 49 to 60 has the lowest number of people in that class.
# 

# In[14]:


class_labels = {1: 'First Class', 2: 'Second Class', 3: 'Third Class'}
for class_val, data in dataset.groupby('Pclass')['Age']:
    plt.hist(data, alpha=.5, label=class_labels[class_val])
plt.title("Age Distribution By Class")
plt.xlabel('Age')
plt.ylabel("Frequency")
plt.legend()


# 7.  FARE DISTRIBUTION

# Below discribe the fare distribution.
# the highest fare paid was about 380 in number which has the more number of persons followed by 200 to 300 while 100 to 200 has the smaller number of people that paid

# In[15]:


dataset['Fare'].hist(bins=5, grid = False);


# 8.  FARE DISTRIBUTION BY CLASS

# The below chart shows the distribution of by each of the class.
# The class are being represented  by color just as we saw in the age distribution by class chart above.
# The third class people where the people with the highest  number of people that paid of about 180 in number and with the amount that range from 0 to 80.
# 
# Followed by the First class which their fare was between  0 to 100 with the highest number of about 75 and people that paid 100 to 300 has the smaller  number of persons
# 
# Person's  in the secondclass has the lowest number which is just 50 in number and their fare amount is between 0 to 90.

# In[19]:


class_labels = {1: 'First Class', 2: 'Second Class', 3: 'Third Class'}
for class_val, data in dataset.groupby('Pclass')['Fare']:
    plt.hist(data, alpha=0.5, bins=5, label=class_labels[class_val])
plt.title("Fare Distribution By Class")
plt.xlabel('Fare')
plt.ylabel("Frequency")
plt.legend()
plt.show()


# 9.  DISTRIBUTION OF PASSENGERS THAT EMBARKED

# The below belows shows the distribution of passengers and Embarked.
# Best on the analysis and the visualization people in the QEmbarked had the highest  no of people of about (270)  followed with the CEmbarked with the total number of (102) while the SEmbarked comes last with the total of (46).

# In[24]:


passengers_by_port = dataset['Embarked'].value_counts()
passengers_by_port


# In[28]:


passengers_by_port.plot(kind ='bar')
plt.title("distribution of passangers & Emberked");


# 10. ANALYSISING NAMES AND TITLE

# In the analysis below we can see the number of people with their title that Embarked on the Tatiana althought  we lost all the men aoccurring to the dataset.

# In[29]:


dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_survival_correlation = dataset.groupby('Title')['Survived'].mean()
title_survival_correlation


# In[30]:


dataset['Title'].value_counts()


# 11.  CORRELATION

# The heatmap below shows that there is a nagetive correlation. and for us to know we also get the number printed normal insead of the scientific format to may thing easier for understanding.

# In[37]:


import seaborn as sns
plt.figure(figsize = (12,8))
sns.heatmap(dataset[['Survived', 'Pclass', 'Age', 'SibSp',
   'Parch', 'Fare']].corr(), annot = True, cmap = 'viridis')


# CONCUSSION 

# In the above analysis of the Titanic shipreck  we have being able to derive insight from the Data set by knowing the number of people that Embarked, the number of presons that survive the ship break, the number of people in each class,the amount of fare that was paid per class and the number of persons that paid each amount, how many paid a particular amount and the age range.
# We are also able to identify the pattern in the data set which people in a particular age range intense to do things the same and we have also been able to identify the corrilation there have between each other.

# In[ ]:




