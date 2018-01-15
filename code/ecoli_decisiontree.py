from scipy.io import arff
import numpy as np
from sklearn import tree
from sklearn import metrics
from pandas import DataFrame
"""Building classifier code """



def loadARFF(filename):
    data, meta = arff.loadarff(filename)
    df = DataFrame(data=data, columns=meta.names())
    y = df["Class"].map({b'positive': 1, b'negative': 0})
    X = df.drop(["Class"], axis=1)
    return (X,y)


X_train,y_train = loadARFF("../datasets/ecoli2-5.46/ecoli2-5-1tra.dat")

X_test,y_test = loadARFF("../datasets/ecoli2-5.46/ecoli2-5-1tst.dat")

dtree = tree.DecisionTreeClassifier();
dtree.fit(X_train.values,y_train.values);


y_predicted = dtree.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_predicted)
precision = metrics.precision_score(y_test, y_predicted)
recall = metrics.recall_score(y_test, y_predicted)
matrix = metrics.confusion_matrix(y_test,y_predicted)

print(matrix)