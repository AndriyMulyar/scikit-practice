from scipy.io import arff
import numpy as np
from sklearn import tree
from sklearn import metrics
from pandas import DataFrame
import graphviz
"""Building classifier code """



def loadARFF(filename):
    data, meta = arff.loadarff(filename)
    df = DataFrame(data=data, columns=meta.names())
    y = df["Class"].map({b'positive': 1, b'negative': 0})
    X = df.drop(["Class"], axis=1)
    return (X,y,meta)


folds = 5;

avg_precision = 0;
avg_recall = 0;

for fold in range(1,folds+1):
    X_train, y_train, meta_train = loadARFF("../datasets/ecoli2-5.46/ecoli2-5-%stra.dat" % fold)

    X_test, y_test, meta_test = loadARFF("../datasets/ecoli2-5.46/ecoli2-5-%stst.dat" % fold)

    dtree = tree.DecisionTreeClassifier();
    dtree.fit(X_train.values, y_train.values);

    y_predicted = dtree.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_predicted)
    precision = metrics.precision_score(y_test, y_predicted, average=None)
    recall = metrics.recall_score(y_test, y_predicted, average=None)
    matrix = metrics.confusion_matrix(y_test, y_predicted)

    avg_precision += precision[0];
    avg_recall += recall[0];

    print("Fold %i" % fold)
    print(matrix)
    print("Accuracy: %f" % accuracy)
    print("Precision: %f" % precision[0])
    print("Recall: %f" % recall[0] ,)

avg_precision /= folds
avg_recall /= folds
print("\nAverage Precision: %f" % (avg_precision))
print("Average Recall: %f" % (avg_recall))