#importing modules
import numpy as np
import pandas as pd
from sklearn import tree
import pickle

#reading and modifing the dataframe features
train = pd.read_csv('train.csv')
train.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)
train['Sex'].replace({'male': 0, 'female': 1}, inplace=True)

train['Family'] = train['SibSp'] + train['Parch']

train.drop(['SibSp', 'Parch'], axis=1, inplace=True)

#constructing features and label dataframe
features = train.drop('Survived', axis=1)
label = train['Survived']


#finding mean of age to fill NaN values in dataframe
age = features['Age'].mean()
features['Age'].fillna(age, inplace=True)


#classification and model is stored in model
cla = tree.DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model = cla.fit(features, label)


#below lines are used to store the model as pickle
#f = open('model.pkl', 'wb')
#pickle.dump(model, f)
#now predicting
#f = open('model.pkl', 'rb')
#cla = pickle.load(f)


#including test.csv and constructing dataframe features
test = pd.read_csv('test.csv')
test.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)
test['Sex'].replace({'male': 0, 'female': 1}, inplace=True)
test['Family'] = df['SibSp'] + df['Parch']
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test['Age'].fillna(m, inplace=True)


#predicting and storing the result in my_subission.csv
compare = pd.DataFrame()
compare['Survived'] = cla.predict(test)
compare['PassengerId'] = test['PassengerId']
compare.to_csv('my_submission.csv', sep=',', encoding='utf-8')
