
...welcome back!


#TITANIC Part II

##Machine Learning from Disaster: Introduction to Scikit-Learn


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.io.sql as pd_sql
import sqlite3 as sql

%matplotlib inline
```


```python
# re-establish database connection
con = sql.connect("XXXX.db") # replace XXXX with whatever you called your titanic database last time

# extract everything from the training data table into a dataframe
train = pd_sql.read_sql('select * from XXXX', con, index_col='index') # replace XXXX with whatever you called your table 
```


```python
# Double check to see if it's all still there!
train.head(5)
```

### Creating a Model: A Naive Approach    

Let's start by creating a model that predicts purely based on gender


```python
# Set some variables
number_passengers = train.shape[0] 
number_survived = len(train[train.Survived == 1])

# What proportion of the passengers survived?
proportion_survived = float(number_survived) / number_passengers
print 'The proportion of passengers who survived is %s.' % proportion_survived
```


```python
# How can we determine what proportion of the women and of the men who survived?
# Let's start by segregating the men and women
women = train[train.Sex == "female"]
men = train[train.Sex != "female"]

# Determine the proportion of women who survived
proportion_women_survived = float(len(women[women.Survived == 1])) / len(women)
print 'The proportion of women who survived is %s.' % proportion_women_survived

# Determine the proportion of men who survived
proportion_men_survived = float(len(men[men.Survived == 1])) / len(men)
print 'The proportion of men who survived is %s.' % proportion_men_survived
```

So we know that women were MUCH more likely to survive, and we could just say that our model is:
- if female => survived = 1
- if male => survived = 0

But that means our predictions are going to be wrong sometimes -- for about a quarter of the women and a fifth of the men. Let's use the Python library Scikit-learn to see if we can do a little better!

## Using Scikit-learn

Scikit-Learn is a powerful machine learning library implemented in Python with numeric and scientific computing powerhouses Numpy, Scipy, and matplotlib for extremely fast analysis of small to medium-sized data sets. It is open source, commercially usable and contains many modern machine learning algorithms for classification, regression, clustering, feature extraction, and optimization. For this reason Scikit-Learn is often the first tool in a data scientist's toolkit for machine learning of incoming data sets.

Scikit-learn will expect numeric values and no blanks, so first we need to do a bit more wrangling.


```python
# 'Sex' is stored as a text value. We should convert (or 'map') it into numeric binaries 
# so it will be ready for scikit-learn.
train['Sex'] = train['Sex'].map({'male': 0,'female': 1})
```


```python
# Scikit-learn won't be tolerant of the missing values. In the last class, we dropped
# the 'Ticket' column. Let's also drop the 'Cabin' and 'Embarked' columns
train = train.drop(['Cabin'], axis=1)
train = train.drop(['Embarked'], axis=1)
```


```python
# Let's also drop the 'Name' column for now (though I can think of some interesting 
# data that might be embedded in those salutations...)
train = train.drop(['Name'], axis=1)
```

Ok, we've got a table of purely numeric data with no null values. We're ready to go.

### LOGISTIC REGRESSION

A logistic regression mathematically calculates the decision boundary between the possibilities. It looks for a straight line that represents a cutoff that most accurately represents the training data.


```python
from sklearn.linear_model import LogisticRegression
```


```python
# Load the test data
test = pd.read_csv("../titanic/data/test.csv") 
test["Age"] = test["Age"].fillna(train["Age"].median())

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

test = test.drop(['Cabin'], axis=1)
test = test.drop(['Embarked'], axis=1)
test = test.drop(['Name'], axis=1)
test = test.drop(['Ticket'], axis=1)
```


```python
# Initialize our algorithm
lr = LogisticRegression(random_state=1)
```


```python
# Define our predictors
predictors = ["Pclass", "Sex", "Age", "SibSp"]
expected  = train["Survived"]

# Train the algorithm using all the training data
lr.fit(train[predictors], expected)

# Make predictions using the training set -- where we already know the correct answers
predicted = lr.predict(train[predictors])
```


```python
# Make predictions based on the test data
predictions = lr.predict(test[predictors])

# Frame your submission for Kaggle
test_predictions = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
test_predictions.head(10)
```

Well that was easy! But how can we find out how well it worked?

### A Word on Cross-Validation   

For Kaggle, the training samples are constructed by splitting our original dataset into more than one part. But what if certain chunks of our data have more variance than others? We want to ensure that our model performs just as well regardless of the particular way the data are divided up. So let's go back and do some cross-validation splits.    

More on cross-validation tools inside Scikit-learn here:    
http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html


```python
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
```


```python
X = train[["Pclass", "Sex", "Age", "SibSp"]]
y = train["Survived"]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)
log_reg = lr.fit(X_train, y_train)
```


```python
# Every estimator has a score method that can judge the quality of the 
# fit (or the prediction) on new data. Bigger is better.   
log_reg.score(X_test, y_test)
```

We can also ask for a classification report.


```python
from sklearn.metrics import classification_report
```


```python
expected   = y_test
predicted  = log_reg.predict(X_test)
classificationReport = classification_report(expected, predicted, target_names=["Perished","Survived"])
print classificationReport
```

How do I interpret this report?
    
Precision is the number of correct positive results divided by the number of all positive results (e.g. how many of the passengers we predicted would survive actually did survive?).

Recall is the number of correct positive results divided by the number of positive results that should have been returned (e.g. how many of the passengers who did survive did we accurately predict would survive?). 

The F1 score is a measure of a test's accuracy. It considers both the precision and the recall of the test to compute the score. The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst at 0.    

    precision = true positives / (true positives + false positives)

    recall = true positives / (false negatives + true positives)

    F1 score = 2 * ((precision * recall) / (precision + recall))
    
So how well did our Logistic Regression Model do?    


```python
# Make predictions based on the test data
predictions = log_reg.predict(test[predictors])

# Frame your 2nd submission to Kaggle
kgl_submission_lr = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
kgl_submission_lr.to_csv('lr_model.csv', index=False)
```

### RANDOM FOREST    

Some models will work better than others! Let's try another one.    

A random forest is a 'meta estimator'. It will fit a number of decision trees (we'll have to tell it how many) on various sub-samples of the dataset. Then it will use averaging to improve the predictive accuracy and control over-fitting.    

Read more about Random Forests here:    
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
# We'll select 50 trees and opt for 'out-of-bag' samples to estimate the generalization error.
rf = RandomForestClassifier(n_estimators=50, oob_score=True) 
```


```python
# Next split up the data with the 'train test split' method in the Cross Validation module
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# ...and then run the 'fit' method to build a forest of trees
rf.fit(X_train, y_train)
```


```python
rf.score(X_test, y_test)
```


```python
expected   = y_test
predicted  = rf.predict(X_test)
classificationReport = classification_report(expected, predicted, target_names=["Perished","Survived"])
print classificationReport
```

So how did we do with our Random Forest Classifier? Sometimes visualizations can
help us to interpret our results. Here is a function that will take our classification
report and create a color-coded heat map that tells us where our model is strong (deep
reds) and/or weak (lighter pinks).


```python
def plot_classification_report(cr, title='Classification report', cmap=plt.cm.Reds):

    lines = cr.split('\n')
    classes = []
    plotMat = []

    for line in lines[2 : (len(lines) - 3)]:
        t = line.split()
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        plotMat.append(v)

    fig, ax = plt.subplots(1)
    fig = plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    
    for c in range(len(plotMat)+1):
        for r in range(len(classes)):
            try:
                txt = plotMat[r][c]
                ax.text(c,r,plotMat[r][c],va='center',ha='center')
            except IndexError:
                pass
            
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
```


```python
plot_classification_report(classificationReport)
```


```python
# Make predictions based on the test data
predictions = rf.predict(test[predictors])

# Frame your 3rd submission to Kaggle
kgl_submission_rf = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
kgl_submission_rf.to_csv('rf_model.csv', index=False)
```

### SVM    

Support vector machines use points in transformed problem space to separate the classes into groups.


```python
from sklearn.svm import SVC
```


```python
kernels = ['linear', 'poly', 'rbf']

splits     = cross_validation.train_test_split(X,y, test_size=0.2)
X_train, X_test, y_train, y_test = splits

for kernel in kernels:
    if kernel != 'poly':
        model      = SVC(kernel=kernel)
    else:
        model      = SVC(kernel=kernel, degree=3)

model.fit(X_train, y_train)
expected   = y_test
predicted  = model.predict(X_test)

SVC_report = classification_report(expected, predicted)
```


```python
plot_classification_report(SVC_report)
```


```python
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, predicted)  
decision_threshold = np.append(thresholds, 1)

plt.plot(decision_threshold, precision, color='red')  
plt.plot(decision_threshold, recall, color='blue')  
leg = plt.legend(('precision', 'recall'), frameon=True)  
leg.get_frame().set_edgecolor('k')  
plt.xlabel('decision_threshold')  
plt.ylabel('%')  
plt.show
```


```python
# Make predictions based on the test data
predictions = model.predict(test[predictors])

# Frame your 4th submission to Kaggle
kgl_submission_svm = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
kgl_submission_svm.to_csv('svm_model.csv', index=False)
```

So of the four methods - our naive model, as well as Logistic Regression, Random Forest, and SVM, which one performed the best on this data set?

### Resources and Further Reading    

This tutorial is based on the Kaggle Competition, "Predicting Survival Aboard the Titanic"    
https://www.kaggle.com/c/titanic    

As well as the following tutorials:    
https://www.kaggle.com/mlchang/titanic/logistic-model-using-scikit-learn/run/91385    
https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests    
https://github.com/savarin/pyconuk-introtutorial/tree/master/notebooks    

See also:    
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html    
http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html    
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html    
http://scikit-learn.org/stable/modules/svm.html      


```python

```
