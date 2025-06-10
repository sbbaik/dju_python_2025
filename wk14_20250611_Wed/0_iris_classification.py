import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def run(x_train, y_train, x_test, y_test, clf):
    clf.fit(x_train, y_train)
    print("    predictions  :", clf.predict(x_test))
    print("    actual labels:", list(y_test))
    print("    score = %0.4f" % clf.score(x_test, y_test))
    print()

def main():
    # fetch dataset using ucimlrepo
    iris = fetch_ucirepo(id=53)
    X = iris.data.features.values        # pandas DataFrame → numpy array
    y = iris.data.targets['class'].values  # target column 이름은 'class'

    # train-test split (기존 방식처럼 120:30 분할)
    x_train, x_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]

    # 증강 데이터 대체 없음 → 동일 데이터 사용
    xa_train, ya_train = x_train, y_train
    xa_test, ya_test = x_test, y_test

    print("Nearest centroid:")
    run(x_train, y_train, x_test, y_test, NearestCentroid())
    print("k-NN classifier (k=3):")
    run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=3))
    print("Naive Bayes classifier (Gaussian):")
    run(x_train, y_train, x_test, y_test, GaussianNB())
    print("Naive Bayes classifier (Multinomial):")
    run(x_train, y_train, x_test, y_test, MultinomialNB())
    print("Decision Tree classifier:")
    run(x_train, y_train, x_test, y_test, DecisionTreeClassifier())
    print("Random Forest classifier (estimators=5):")
    run(xa_train, ya_train, xa_test, ya_test, RandomForestClassifier(n_estimators=5))

    print("SVM (linear, C=1.0):")
    run(xa_train, ya_train, xa_test, ya_test, SVC(kernel="linear", C=1.0))
    print("SVM (RBF, C=1.0, gamma=0.25):")
    run(xa_train, ya_train, xa_test, ya_test, SVC(kernel="rbf", C=1.0, gamma=0.25))
    print("SVM (RBF, C=1.0, gamma=0.001, augmented)")
    run(xa_train, ya_train, xa_test, ya_test, SVC(kernel="rbf", C=1.0, gamma=0.001))
    print("SVM (RBF, C=1.0, gamma=0.001, original)")
    run(x_train, y_train, x_test, y_test, SVC(kernel="rbf", C=1.0, gamma=0.001))

main()
