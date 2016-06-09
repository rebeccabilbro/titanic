
# TITANIC: Wrangling the Passenger Manifest

## Exploratory Analysis with Pandas

This tutorial is based on the Kaggle Competition,
"Predicting Survival Aboard the Titanic"
https://www.kaggle.com/c/titanic

___Be sure to read the README before you begin!___

See also:    
http://www.analyticsvidhya.com/blog/2014/08/baby-steps-python-performing-exploratory-analysis-python/    
http://www.analyticsvidhya.com/blog/2014/09/data-munging-python-using-pandas-baby-steps-python/


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.io.sql as pd_sql
import sqlite3 as sql

%matplotlib inline
```

Here's a ```sqlite``` database for you to store the data once it's ready:


```python
con = sql.connect("titanic.db") 
```

__=>YOUR  TURN!__

Use ```pandas``` to open up the csv.

Read the documentation to find out how:
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html


```python
# Use pandas to open the csv. 
# You'll have to put in the filepath
# It should look something like "../titanic/data/train.csv"

df = 
```

### Exploring the Tabular Data

The file we'll be exploring today, ```train.csv```, is the training set -- it represents
a subset of the full passenger manifest dataset. The rest of the data is in another
file called ```test.csv``` - we'll use that later (when we get to Machine Learning).
Let's take a look...

__=>YOUR  TURN!__

Use ```pandas``` to view the "head" of the file with the first 10 rows.

Read the documentation to find out how:    
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html


```python
# Use pandas to view the first 10 rows.
```

__What do you see?__
    - Are there any missing values?
    - What kinds of values/numbers/text are there?
    - Are the values continuous or categorical?
    - Are some variables more sparse than others?
    - Are there multiple values in a single column?

__=>YOUR  TURN!__

Use ```pandas``` to run summary statistics on the data.

Read the documentation to find out how:    
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html


```python
# Use pandas to get the summary statistics.
```

__What can we infer from the summary statistics?__
    - How many missing values does the 'Age' column have?
    - What's the age distribution?
    - What percent of the passengers survived?
    - How many passengers belonged to Class 3?
    - Are there any outliers in the 'Fare' column?

__=>YOUR  TURN!__

Use ```pandas``` to get the median for the Age column.

Read the documentation to find out how:    
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.median.html


```python
# Use pandas to get the median age.
```

__=>YOUR  TURN!__

Use ```pandas``` to find the number of unique values in the Ticket column.

Read the documentation to find out how:    
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.nunique.html


```python
# Use pandas to count the number of unique Ticket values.
```

### Visually Exploring the Data

Let's look at a histogram of the age distribution.
What can you tell from the graph?


```python
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(df['Age'], bins = 10, range = (df['Age'].min(),df['Age'].max()))
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Count of Passengers')
plt.show()
```

Now let's look at a histogram of the fares.
What does it tell you?


```python
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(df['Fare'], bins = 10, range = (df['Fare'].min(),df['Fare'].max()))
plt.title('Fare distribution')
plt.xlabel('Fare')
plt.ylabel('Count of Passengers')
plt.show()
```

### Dealing with Missing Values

Part of data wrangling is figuring out how to deal with missing values.
But before you decide, think about which variables are likely to be predictive
of survival. Which ones do you think will be the best predictors?

__Age__
Age is likely to play a role, so we'll probably want to estimate or 'impute'
the missing values in some way.

__Fare__
There are a lot of extremes on the high end and low end for ticket fares.
How should we handle them?

__Other Variables__
What do YOU think??

__=>YOUR  TURN!__

Use ```pandas``` to get the sum of all the null values in the Cabin column.

Read the documentation to find out how:    
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.isnull.html    
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sum.html


```python
# Use pandas to sum the null Cabin values.
```

__=>YOUR  TURN!__

Use ```pandas``` to drop the Ticket column.

Read the documentation to find out how:    
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html


```python
# Use pandas to drop the Ticket column.
```

__=>YOUR  TURN!__

Use ```pandas``` to calculate the mean age and fill all the null values in the Age column with that number..

Read the documentation to find out how:    
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mean.html     
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html


```python
# Use pandas to get the mean Age.
# Use pandas to fill in the null Age values with the mean.
```

### Save Your Work
...you will need it in a few weeks!

__=>YOUR  TURN!__

Use ```pandas``` to write your dataframe to our sqlite database.

Read the documentation to find out how:   
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html


```python
# Use pandas to save your dataframe to a sqlite database.
```
