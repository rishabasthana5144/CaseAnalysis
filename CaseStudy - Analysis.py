# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 23:35:30 2021

@author: asus
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
sns.set_style(style='whitegrid')


# Plots the charts on seperate window
%matplotlib qt 

os.chdir('C:/Users/asus/Downloads/CaseStudy')

# Reading data
df = pd.read_csv("Churn Modeling.csv")

# Dropping useless columns
df = df.drop(['RowNumber','CustomerId','Surname'], axis = 'columns')

# Checking data types
df.dtypes

# Checking head of the data
df.head()


# Distribution of Credit Score
sns.distplot(a = df.CreditScore, hist = True,  kde = False) 
plt.xlabel('Credit Score', fontsize = 20)
plt.ylabel('Count of Customers',fontsize = 20)
plt.title('Distribution of Credit Score', fontsize = 20)
plt.tick_params(axis = 'both', labelsize=20)
plt.legend(loc = 'upper left', fontsize = 20)
plt.show()
##


# Distribution of Credit Score by target
sns.distplot(a = df.loc[df['Exited'] == 1,].CreditScore, hist = False) 
sns.distplot(a = df.loc[df['Exited'] == 0,].CreditScore, hist = False) 
plt.xlabel('Credit Score', fontsize = 20)
plt.ylabel('Density',fontsize = 20)
plt.title('Distribution of Credit Score Over Attrition', fontsize = 20)
plt.tick_params(axis = 'both', labelsize=20)
plt.legend(loc = 'upper left', fontsize = 20)
plt.show()


# Distribution of Credit Score by Attrition
sns.boxplot(x = df['Exited'],
            y = df['CreditScore'],
            palette = 'Set2')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 25}

plt.title("Distribution of Credit Score over Attrition", fontdict=font)
#plt.legend(s=10)
plt.yticks(fontsize = 25)
plt.ylabel("Credit Score", fontdict=font)
plt.xlabel("Exited", fontdict=font)
plt.xticks(fontsize = 25)



# Gender distribution
gn_df = pd.DataFrame(df['Gender'].value_counts()).reset_index().rename(columns = {'index': 'Gender', 'Gender': 'Count'})
gn_df['Percent'] = gn_df['Count']/sum(gn_df['Count'])
gn_df

plt.bar(x = gn_df.Gender, height = gn_df.Percent, align = 'center')
font = {'family': 'serif', 'color':  'darkred', 'weight': 'normal','size': 16}

plt.title("Gender Distribution", fontdict=font)
plt.xlabel("Gender", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count", fontdict=font)
plt.xticks(fontsize = 15)

plt.barh(y = gn_df.Gender, width= gn_df.Count)

# gender distirbution over target variable
sns.countplot(df['Gender'])
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Gender", fontdict=font)
plt.xlabel("Gender", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)




# Distribution of Gender by Attrition
sns.countplot(x='Gender', hue = 'Exited',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Gender distribution over Attrition", fontdict=font)
plt.xlabel("Gender", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Customer Count", fontdict=font)
plt.xticks(fontsize = 15)

# The proportions of female churning is more than males


# Distribution of Age 
sns.distplot(df.Age, hist = True)
sns.boxplot(df.Age)
# Fair distirbution of Age noticed. However outlier noticed in the age

#Distirbution of Age by attrition
sns.displot(data=df, x='Age', hue='Exited', kind='kde')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 25}
plt.title("Age Distirbution over Attrition", fontdict=font)
plt.xlabel("Age", fontdict=font)
plt.yticks(fontsize = 25)
plt.ylabel("Density", fontdict=font)
plt.legend(loc = 'upper right')
plt.xticks(fontsize = 25)


# Distribution of Number of Products by Estimated Salary
sns.boxplot(x = df['NumOfProducts'],
            y = df['EstimatedSalary'],
            palette = 'Set2')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Age over Attrition", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Age", fontdict=font)
plt.xticks(fontsize = 15)
# Slightly higher age where attrition is occuring

# Analysis of Tenure
sns.boxplot(x = df['Exited'],
            y = df['Tenure'],
            palette = 'Set2')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Tenure over Attrition", fontdict=font)
#plt.legend(s=10)
plt.yticks(fontsize = 25)
plt.ylabel("Tenure", fontdict=font)
plt.xticks(fontsize = 25)


# Analysis of Balance over Exited
sns.boxplot(x = df['Exited'],
            y = df['Balance'],
            palette = 'Set2')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Balance over Attrition", fontdict=font)
#plt.legend(s=10)
plt.yticks(fontsize = 15)
plt.ylabel("Balance", fontdict=font)
plt.xticks(fontsize = 15)



# Geography distribution
sns.countplot(df.Geography)
# 50 percent of population is from West


# Distribution of attrition over geography
sns.countplot(x='Geography', hue = 'Exited',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("Geography distribution over Attrition", fontdict=font)
plt.xlabel("Regions", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Density", fontdict=font)
plt.xticks(fontsize = 15)

# Lesser the customers with the bank, higher the number of churned customers are.
# Compromised Customer Service?


# Distribution of Number of Products by Attrition
sns.countplot(x='NumOfProducts',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Number of Products", fontdict=font)
#plt.legend(s=10)
plt.xlabel("Number of products", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count", fontdict=font)
plt.xticks(fontsize = 15)



sns.countplot(x='NumOfProducts', hue = 'Exited',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Number of Products over Attrition", fontdict=font)
#plt.legend(s=10)
plt.xlabel("Number of products", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count", fontdict=font)
plt.xticks(fontsize = 15)
# Number of Products key a key predictor
# Customers with single product leaves the most
# Product Count 2 does not show much attrition

# HasChckng over Exited
sns.countplot(x='HasChckng', hue = 'Exited',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("HasChckng over Attrition", fontdict=font)
#plt.legend(s=10)
plt.xlabel("HasChckng", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)
#plt.savefig('HasChckng over Attrition - hist')


# Is Active member over Exited
sns.countplot(x='IsActiveMember', hue = 'Exited',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Digitally Active over Attrition", fontdict=font)
#plt.legend(s=10)
plt.xlabel("Digitally Active", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)


# Gender over Exited
sns.countplot(x='Gender', hue = 'Exited',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("HasChckng over Attrition", fontdict=font)
#plt.legend(s=10)
plt.xlabel("HasChckng", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)




# Analysis of Estimated Salary 

# Over Region
sns.boxplot(x = df['Geography'],
            y = df['EstimatedSalary'])

# Over Exited
sns.boxplot(x = df['Exited'],
            y = df['EstimatedSalary'],
            palette = 'Set2')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Distribution of Salary over Attrition", fontdict=font)
#plt.legend(s=10)
plt.yticks(fontsize = 15)
plt.ylabel("Estimated Salary", fontdict=font)
plt.xticks(fontsize = 15)



# Target Distribution
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18
labels = 'Exited', 'Unchurned'
sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and unchurned", size = 20)
plt.show()





# Over distribution of Numeric variable
plt.hist(df['CreditScore'])


#######################################
# Univariate Analysis
#######################################
cols = ['Age','Balance', 'CreditScore', 'EstimatedSalary']
color = ['red','green','purple', 'blue']
for index in range(0,4):

    # Calling the subplot
    plt.subplot(2, 2, index+1)
    
    # Plotting chart on called subplot
    sns.distplot(a = df[cols[index]], kde = False, hist = True, color = color[index], hist_kws={'edgecolor' : 'black'})
    
    # Setting the aesthetics
    plt.xlabel(cols[index], fontsize = 15)
    plt.ylabel('Employee Count',fontsize = 15)
    plt.tick_params(axis = 'both', labelsize=15)
    plt.show()


fig, axarr = plt.subplots(2, 2, figsize=(20, 12))

sns.distplot(a = df.Age, hist = True,  kde = False, color='red', ax = axarr[0][]) 
plt.xlabel('Age', fontsize = 20)
plt.ylabel('Count of Customers',fontsize = 20)
plt.title('Distribution of Age', fontsize = 20)
plt.tick_params(axis = 'both', labelsize=20)
plt.legend(loc = 'upper left', fontsize = 20)
plt.show()



# We first review the 'Status' relation with categorical variables
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.histplot(data = df, x = 'Age')
(x='Geography', hue = 'Exited',data = df, ax=axarr[0][0])
sns.countplot(x='Gender', hue = 'Exited',data = df, ax=axarr[0][1])
sns.countplot(x='HasCrCard', hue = 'Exited',data = df, ax=axarr[1][0])
sns.countplot(x='IsActiveMember', hue = 'Exited',data = df, ax=axarr[1][1])

# Continuous variable

fig, ax =  plt.subplots(1, 2, figsize=(15, 7))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.scatterplot(x = "Age", y = "Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax=ax[0])
sns.scatterplot(x = "Age", y = "CreditScore", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax=ax[1])



fig, ax =  plt.subplots(1, 2, figsize=(15, 7))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.histplot(x = df.CreditScore, ax=ax[0])


fig, axes =  plt.subplots(2, 2, figsize=(15, 7))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.distplot(a = df.CreditScore, hist = True,  kde = False, ax=axes[0]) 
plt.xlabel('Credit Score', fontsize = 20)
plt.ylabel('Count of Customers',fontsize = 20)
plt.title('Distribution of Credit Score', fontsize = 20)
plt.tick_params(axis = 'both', labelsize=20)
plt.legend(loc = 'upper left', fontsize = 20)
plt.show()




sns.distplot(a = df.Age, hist = True,  kde = False, color='red') 
plt.xlabel('Age', fontsize = 20)
plt.ylabel('Count of Customers',fontsize = 20)
plt.title('Distribution of Age', fontsize = 20)
plt.tick_params(axis = 'both', labelsize=20)
plt.legend(loc = 'upper left', fontsize = 20)
plt.show()

sns.distplot(a = df.Balance, hist = True,  kde = False) 
plt.xlabel('Balance', fontsize = 20)
plt.ylabel('Count of Customers',fontsize = 20)
plt.title('Distribution of Balance', fontsize = 20)
plt.tick_params(axis = 'both', labelsize=20)
plt.legend(loc = 'upper left', fontsize = 20)
plt.show()
    
    
sns.distplot(a = df.EstimatedSalary, hist = False,  kde = True) 
plt.xlabel('EstimatedSalary', fontsize = 20)
plt.ylabel('Density',fontsize = 20)
plt.title('Distribution of Salary', fontsize = 20)
plt.tick_params(axis = 'both', labelsize=20)
plt.show()


# Categorical variable

sns.countplot(df.Tenure) 
plt.xlabel('Tenure', fontsize = 20)
plt.ylabel('Count of Customers',fontsize = 20)
plt.title('Distribution of Tenure', fontsize = 20)
plt.tick_params(axis = 'both', labelsize=20)
plt.legend(loc = 'upper left', fontsize = 20)
plt.show()


sns.countplot(x='NumOfProducts',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 32}
plt.xlabel("Number of products", fontdict=font)
plt.yticks(fontsize = 32)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 32)


sns.countplot(x='HasChckng',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("HasChckng", fontdict=font)
plt.xlabel("HasChckng", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)

sns.countplot(x='Geography',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("Geography", fontdict=font)
plt.xlabel("Geography", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)

sns.countplot(x='Gender',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("Gender", fontdict=font)
plt.xlabel("Gender", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)


sns.countplot(x='IsActiveMember',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("IsActiveMember", fontdict=font)
plt.xlabel("IsActiveMember", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)



# Bivariate Analysis

# Numerical with Targe
sns.displot(data=df, x='CreditScore', hue='Exited', kind = 'kde')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Credit Score with Attrition", fontdict=font)
plt.xlabel("Credit Score", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Density", fontdict=font)
plt.xticks(fontsize = 15)


sns.boxplot(x = df['Exited'],
            y = df['CreditScore'],
            palette = 'Set2')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Distribution of Credit Score over Attrition", fontdict=font)
#plt.legend(s=10)
plt.yticks(fontsize = 15)
plt.ylabel("Credit Score", fontdict=font)
plt.xticks(fontsize = 15)


# Analysis of Balance over Exited
sns.displot(data=df, x='Balance', hue='Exited', kind = 'kde')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("Balance over Attrition", fontdict=font)
plt.xlabel("Balance", fontdict=font)
plt.yticks(fontsize = 25)
plt.ylabel("Density", fontdict=font)
plt.xticks(fontsize = 25)


sns.boxplot(x = df['Exited'],
            y = df['Balance'],
            palette = 'Set2')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Balance over Attrition", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Balance", fontdict=font)
plt.xticks(fontsize = 15)


# Analysis of Age over Attrition
sns.displot(data=df, x='Age', hue='Exited', kind = 'kde')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("Age with Attrition", fontdict=font)
plt.xlabel("Age", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Density", fontdict=font)
plt.xticks(fontsize = 15)


sns.boxplot(x = df['Exited'],
            y = df['Age'],
            palette = 'Set2')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Age over Attrition", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Age", fontdict=font)
plt.xticks(fontsize = 15)
# Slightly higher age where attrition is occuring



# Analysis of Estimated Salary over Attrition
sns.boxplot(x = df['Exited'],
            y = df['EstimatedSalary'],
            palette = 'Set2')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Distribution of Salary over Attrition", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Estimated Salary", fontdict=font)
plt.xticks(fontsize = 15)



# Categorical with Target


# Distribution of attrition over geography
sns.countplot(x='Geography', hue = 'Exited',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("Geography distribution over Attrition", fontdict=font)
plt.xlabel("Regions", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)


sns.countplot(x='Gender', hue = 'Exited',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("Gender distribution over Attrition", fontdict=font)
plt.xlabel("Gender", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)


sns.countplot(x='NumOfProducts', hue = 'Exited',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("Number of Products over Attrition", fontdict=font)
plt.xlabel("Number of products", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)

sns.countplot(x='HasChckng', hue = 'Exited',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("HasChckng over Attrition", fontdict=font)
plt.xlabel("HasChckng", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)


sns.countplot(x='IsActiveMember', hue = 'Exited',data = df)
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("Active Member over Attrition", fontdict=font)
plt.xlabel("Is Active", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Count of Customers", fontdict=font)
plt.xticks(fontsize = 15)



# Credit Score over Geography
sns.boxplot(x = df['EstimatedSalary'],
            y = df['Exited'],
            palette = 'Set2')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}
plt.title("Credit Score over Geography", fontdict=font)
plt.yticks(fontsize = 15)
plt.ylabel("Geography", fontdict=font)
plt.xticks(fontsize = 15)



# Looking at Geography and Gender Distribution against Estimated Salary
plt.figure(figsize=(20,20))
sns.catplot(x="Geography", y="EstimatedSalary", hue="Gender", kind="box", data=df)
plt.title("Geography VS Estimated Salary")
plt.xlabel("Geography")
plt.ylabel("Estimated Salary")

# Estimated salary is higher for females in Central and East


_, ax =  plt.subplots(1, 2, figsize=(15, 7))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.scatterplot(x = "Age", y = "Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax=ax[0])
sns.scatterplot(x = "Age", y = "CreditScore", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax=ax[1])


cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.scatterplot(x = "Age", y = "Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df.loc[df['Age'] < 95])
plt.yticks(fontsize = 25)
plt.ylabel('Balance', size = 25)
plt.xticks(fontsize = 25)
plt.xlabel('Age', size = 25)
plt.title('Attrition over Balance and Age', size = 25)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.scatterplot(x = "Age", y = "CreditScore", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df.loc[df['Age'] < 95])
plt.yticks(fontsize = 25)
plt.ylabel('Credit Score', size = 25)
plt.xticks(fontsize = 25)
plt.xlabel('Age', size = 25)
plt.title('Attrition over Credit Score and Age', size = 25)


# hypothesis
1. Does lower estimated salary increase churn?
2. Does lower credit score increase churn?
3. As one grows older, credit score improves?

# Looking at linear relationship between Age and CreditScore
plt.figure(figsize=(10,10))
sns.regplot(x="Age", y="CreditScore", data=df)

# Correlation Chart
sns.heatmap(df[['CreditScore','Age','Balance','EstimatedSalary']].corr(), cmap = 'Blues', annot=True)




# Correlation analysis of numeric variables
df[['CreditScore','Age','Balance','EstimatedSalary','Tenure']].corr()
# Weak correlation

# Grouping 
#Grouping Neutral to 'Male': On what basis should we include it?
df.groupby(['Geography','Gender'])['Gender'].size()

df.loc[df['Gender'] == 'Neutral','Gender'] = 'Male'

# Very less records for product count 4 and all of them have exited: should be excluded from the data as it is the case where business would have to defnitely pitch them and they are sure short churned people

# Excluding people with product count 4
df = df[df['NumOfProducts'] != 4]


# Outliers in age, can be imputed with mean using 99 percent method
# Imputing outliers in age - 99 percentile

df.loc[df['Age']>df['Age'].quantile(.99),'Age'] = df.loc[df['Age']<df['Age'].quantile(.99),'Age'].mean()  



# Split into train and test
df_train = df.sample(frac=0.8,random_state=200)
df_test = df.drop(df_train.index)
print(len(df_train))
print(len(df_test))

df_train['Exited'].value_counts()
df_test['Exited'].value_counts()


# Additional Features
df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df_train)
plt.ylim(-1, 5)


# Given that tenure is a 'function' of age, we introduce a variable aiming to standardize tenure over age:
df_train['TenureByAge'] = df_train.Tenure/(df_train.Age)
sns.boxplot(y='TenureByAge',x = 'Exited', hue = 'Exited',data = df_train)
plt.ylim(-1, 1)
plt.show()





'''Lastly we introduce a variable to capture credit score given age to take into account credit behaviour visavis adult life
:-)'''
df_train['CreditScoreGivenAge'] = df_train.CreditScore/(df_train.Age)


df_train['Exited'].value_counts()/len(df_train)


# Converting into one hot encoding
df[['Geography','Gender','IsActivemember','HasChckng']]







cols = ['Age','Balance', 'CreditScore', 'EstimatedSalary']
color = ['red','green','purple', 'blue']
for index in range(0,4):

    # Calling the subplot
    plt.subplot(2, 2, index+1)
    
    # Plotting chart on called subplot
    sns.distplot(a = df[cols[index]], kde = False, hist = True, color = color[index], hist_kws={'edgecolor' : 'black'})
    
    # Setting the aesthetics
    plt.xlabel(cols[index], fontsize = 15)
    plt.ylabel('Employee Count',fontsize = 15)
    plt.tick_params(axis = 'both', labelsize=15)
    plt.show()


cols = ['Geography','Gender', 'HasChckng', 'IsActiveMember','NumOfProducts']
color = ['red','green','purple', 'blue','orange']
for index in range(0,5):

    # Calling the subplot
    plt.subplot(2, 2, index+1)
    
    # Plotting chart on called subplot
#    sns.distplot(a = df[cols[index]], kde = False, hist = True, color = color[index], hist_kws={'edgecolor' : 'black'})
    sns.countplot(df[cols[index]]) 


    # Setting the aesthetics
    plt.xlabel(cols[index], fontsize = 15)
    plt.ylabel('Employee Count',fontsize = 15)
    plt.tick_params(axis = 'both', labelsize=15)
    plt.show()






# Stacked Barplot


import matplotlib.ticker as mtick
active_churn= df1.groupby(['Gender', 'Exited']).size().unstack()
active_churn.rename(columns={0:'Retained', 1:'Exited'}, inplace=True)
colors  = ['#ec838a','#9b9c9a', '#f3babc' , '#4d4f4c']
ax = (active_churn.T*100.0 / active_churn.T.sum()).T.plot(
kind='bar',width = 0.3,stacked = True,rot = 0,figsize = (12,7),
color = colors)
plt.ylabel('Proportion of Customers\n',
horizontalalignment="center",fontstyle = "normal", 
fontsize = 20, fontfamily = "sans-serif")
plt.xlabel('Gender\n',horizontalalignment="center",
fontstyle = "normal", fontsize = 20, 
fontfamily = "sans-serif")
#plt.title('Churn Rate by Products Count\n',
#horizontalalignment="center", fontstyle = "normal", 
#fontsize = "32", fontfamily = "sans-serif")
plt.legend(loc='top right', fontsize = "medium")
plt.xticks(rotation=0, horizontalalignment="center", fontsize =20)
plt.yticks(rotation=0, horizontalalignment="right", fontsize = 20)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.text(x+width/2, 
            y+height/2, 
            '{:.1f}%'.format(height), 
            horizontalalignment='center', 
            verticalalignment='center', fontsize = 15)
ax.autoscale(enable=False, axis='both', tight=False)


sns.boxplot(y='EstimatedSalary',x = 'NumOfProducts',data = df)

font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 30}
plt.title("Salary over Products", fontsize = 30)
plt.yticks(fontsize = 30)
plt.ylabel("Salary", fontsize = 30)
plt.xlabel("Number Of Products", fontsize = 30)
plt.xticks(fontsize = 30)


sns.boxplot(x = df['NumOfProducts'],
            y = df['EstimatedSalary'],
            hue = df['Exited'],
            palette = 'Set2')
font = {'family': 'serif', 'color':  'black', 'weight': 'normal','size': 16}

plt.title("Salary over Products", fontsize = 30)
plt.yticks(fontsize = 30)
plt.ylabel("Salary", fontsize = 30)
plt.xlabel("Number Of Products", fontsize = 30)
plt.xticks(fontsize = 30)


# 
# Looking at linear relationship between Age and CreditScore
plt.figure(figsize=(10,10))
sns.regplot(x="Age", y="CreditScore", data=df[df['Age'] < 95])

# Modelling
# Split Train, test data
df_train = df.sample(frac=0.8,random_state=200)
df_test = df.drop(df_train.index)
print(len(df_train))
print(len(df_test))

df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df_train)
plt.ylim(-1, 5)

df_train['TenureByAge'] = df_train.Tenure/(df_train.Age)
sns.boxplot(y='TenureByAge',x = 'Exited', hue = 'Exited',data = df_train)
plt.ylim(-1, 1)
plt.show()

df_train['CreditScoreGivenAge'] = df_train.CreditScore/(df_train.Age)


# Arrange columns by data type for easier manipulation
continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio', 'TenureByAge','CreditScoreGivenAge']
cat_vars = ['HasChckng', 'IsActiveMember','Geography', 'Gender']
df_train = df_train[['Exited'] + continuous_vars + cat_vars]
df_train.head()


df_train.loc[df_train.HasChckng == 0, 'HasChckng'] = -1
df_train.loc[df_train.IsActiveMember == 0, 'IsActiveMember'] = -1
df_train.head()


# One hot encode the categorical variables
lst = ['Geography', 'Gender']
remove = list()
for i in lst:
    if (df_train[i].dtype == np.str or df_train[i].dtype == np.object):
        for j in df_train[i].unique():
            df_train[i+'_'+j] = np.where(df_train[i] == j,1,-1)
        remove.append(i)
df_train = df_train.drop(remove, axis=1)
df_train.head()

# minMax scaling the continuous variables
minVec = df_train[continuous_vars].min().copy()
maxVec = df_train[continuous_vars].max().copy()
df_train[continuous_vars] = (df_train[continuous_vars]-minVec)/(maxVec-minVec)
df_train.head()


#############################
df['CreditScore'] = np.where(df['CreditScore'].isnull(), df['CreditScore'].mean(), df['CreditScore'])

df['Age'] = np.where(df['Age'] > df['Age'].quantile(0.99), df['Age'].mean(), df['Age'])

df['Gender'] = np.where(df['Gender'] == 'Neutral', 'Male', 'Female')

df = pd.get_dummies(df, drop_first = True)



# importing libraries
import statsmodels.api as sm

df['Age'].describe()

X = df.drop(labels = ['Exited'], axis = 'columns')
Y = df.Exited

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=3)


# Initial Model
model = sm.Logit(y_trainset,X_trainset).fit()

# Printing model summary
model.summary()



