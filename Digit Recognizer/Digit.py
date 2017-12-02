# including headers
import pandas as pd
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors.base import KNeighborsMixin
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import SVC
from sklearn.tree.tree import DecisionTreeClassifier
import pickle

# reading files
df = pd.read_csv('train.csv')

df_test = pd.read_csv('test.csv')

df_submission = pd.DataFrame()
a, _ = df_test.shape
df_submission['ImageId'] = range(1, a + 1)

# creating features and labels
X = df.drop('label', axis=1)
Y = df['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

# trying different models
'''
cls = DecisionTreeClassifier(random_state=42)
cls.fit(X_train,Y_train)
print(cls.score(X_test, Y_test))# score = .8475

cls = RandomForestClassifier(random_state=42)
cls.fit(X_train,Y_train)
print(cls.score(X_test, Y_test)) # score = .9343


cls = MLPClassifier(random_state=42,hidden_layer_sizes=(1000,2000,3000,100))
cls.fit(X_train, Y_train)
print(cls.score(X_test, Y_test))

'''

# classifing
cls = SVC(kernel="poly", random_state=42)
cls.fit(X_train, Y_train)
print(cls.score(X_test, Y_test))

'''
f = open("SVC.pkl","rb")
cls = pickle.load(f)
'''

# creating submission file
df_submission_prediction = cls.predict(df_test)
df_submission['Label'] = df_submission_prediction

df_submission.to_csv("mySubmission.csv")

'''
f = open("SVC.pkl","wb")
pickle.dump(cls, f)
f.close()
'''
