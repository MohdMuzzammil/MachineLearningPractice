import pandas as pd
from sklearn.cluster.k_means_ import KMeans
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm.classes import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree.tree import DecisionTreeClassifier

df = pd.read_csv('mushrooms.csv')
columns = df.columns.values


for column in columns:
    le = LabelEncoder()
    le.fit(df[column])
    df[column] = le.transform(df[column])


X = df.drop('class',axis=1)
Y = df['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


cls = SVC(kernel="poly")
cls.fit(X_train, Y_train)
print("SVC",cls.score(X_test, Y_test))# score = 1.0


cls = LinearRegression()
cls.fit(X_train, Y_train)
print("LR",cls.score(X_test, Y_test)) # score = 0.73

cls = LogisticRegression()
cls.fit(X_train, Y_train)
print("LogR",cls.score(X_test, Y_test)) # score = 0.93

cls = DecisionTreeClassifier()
cls.fit(X_train, Y_train)
print("DecTree",cls.score(X_test, Y_test)) # score = 1.0

cls = RandomForestClassifier()
cls.fit(X_train, Y_train)
print("RandomForest",cls.score(X_test, Y_test)) # score = 1.0



