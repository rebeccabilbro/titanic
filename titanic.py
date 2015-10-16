# titanic.py
# This tutorial is based on the Kaggle Competition,
# "Predicting Survival Aboard the Titanic"
# https://www.kaggle.com/c/titanic
#
# See also:
# http://www.analyticsvidhya.com/blog/2014/08/baby-steps-python-performing-exploratory-analysis-python/
# http://www.analyticsvidhya.com/blog/2014/09/data-munging-python-using-pandas-baby-steps-python/

# STEP ONE: EXPLORATORY ANALYSIS

########################################################################
# Imports
########################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.io.sql as pd_sql
import sqlite3 as sql

########################################################################
# Globals
########################################################################
# con = sql.connect("titanic.db") # here we create a database for the clean data
# df = pd.read_csv("../titanic/data/train.csv") # here we use pandas to load the csv in

########################################################################
# Exploring the Tabular Data
########################################################################
'''
The file we'll be exploring today, 'train.csv', is the training set -- it represents
a subset of the full passenger manifest dataset. The rest of the data is in another
file called 'test.csv' - we'll use that later (when we get to Machine Learning).
Let's take a look...
'''
# print df.head(10) # This shows you the first 10 rows.
'''
What do you see?
-Are there any missing values?
-What kinds of values/numbers/text are there?
-Are the values continuous or categorical?
-Are some variables more sparse than others?
-Are there multiple values in a single column?
'''
# print df.describe() # This will give you some summary statistics.
'''
What can we infer from the summary statistics?
-How many missing values does the 'Age' column have?
-What's the age distribution?
-What percent of the passengers survived?
-How many passengers belonged to Class 3?
-Are there any outliers in the 'Fare' column?
'''
# print df['Age'].median() # gives the median age
# print df['Sex'].unique() # gives unique values for the 'Sex' column


########################################################################
# Visually Exploring the Data
########################################################################
'''
Let's look at a histogram of the age distribution.
What can you tell from the graph?
'''
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.hist(df['Age'], bins = 10, range = (df['Age'].min(),df['Age'].max()))
# plt.title('Age distribution')
# plt.xlabel('Age')
# plt.ylabel('Count of Passengers')
# plt.show()
'''
Now let's look at a histogram of the fares.
What does it tell you?
'''
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.hist(df['Fare'], bins = 10, range = (df['Fare'].min(),df['Fare'].max()))
# plt.title('Fare distribution')
# plt.xlabel('Fare')
# plt.ylabel('Count of Passengers')
# plt.show()


########################################################################
# Dealing with Missing Values
########################################################################
'''
Part of data wrangling is figuring out how to deal with missing values.
But before you decide, think about which variables are likely to be predictive
of survival. Which ones do you think will be the best predictors?

Age
Age is likely to play a role, so we'll probably want to estimate or 'impute'
the missing values in some way.

Fare
There are a lot of extremes on the high end and low end for ticket fares.
How should we handle them?

Other Variables
What do YOU think??
'''
# print sum(df['Cabin'].isnull()) # You can check the number of null values.
# df = df.drop(['Ticket'], axis=1) # You can drop columns like this.
# mean_age = np.mean(df.Age) # You can calculate the mean of the column values.
# df.Age = df.Age.fillna(mean_age) # You can fill in null values with the mean.


########################################################################
# Save Your Work... you will need it in a few weeks!
########################################################################
# pd_sql.to_sql(df, "training", con) # write your dataframe to a sqlite database
