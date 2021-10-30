import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load Data
cols = ['ID', 'Thickness', 'U_Size', 'U_Shape', 'Adhersion', 'S_Size', 'Bare', 'Bland', 'Normal', 'Mitoses', 'Class']
data = pd.read_csv('breast-cancer-wisconsin.txt', sep=',', encoding='cp949', names=cols)

# print(data.head())
# print(data.describe())
# print(data.info())

data = data[data.Bare != '?']

# Detach Target feature
X = data.drop(columns='Class')
y = data['Class']



# function label encoding
# input target column list and dataframe
def lblEncoding(listObj, x):
    lbl = preprocessing.LabelEncoder()

    for i in range(len(listObj)):
        x[listObj[i]] = lbl.fit_transform(x[listObj[i]])
    # output encoded dataframe
    return x


# function ordinal encoding
# input target column list and dataframe
def ordEncoding(listObj, x):
    ord = preprocessing.OrdinalEncoder()

    for i in range(len(listObj)):
        tempColumn = x[listObj[i]].to_numpy().reshape(-1, 1)
        tempColumn = ord.fit_transform(tempColumn)
        tempColumn = tempColumn.reshape(1, -1)[0]
        x[listObj[i]].replace(x[listObj[i]].tolist(), tempColumn, inplace=True)
    # output encoded dataframe
    return x


# Decision Tree Entropy
def dteClassifier(X_train, Y_train, X_test, Y_test):
    dte = DecisionTreeClassifier(criterion="entropy")
    dte.fit(X_train, Y_train)
    print(dte.score(X_test, Y_test))


# Decision Tree Gini
def dtgClassifier(X_train, Y_train, X_test, Y_test):
    dtg = DecisionTreeClassifier(criterion="gini")
    dtg.fit(X_train, Y_train)
    print(dtg.score(X_test, Y_test))


# Logistic Regression
def logisticRegr(X_train, y_train, X_test, y_test):
    logisticRegr = LogisticRegression(solver='lbfgs')
    logisticRegr.fit(X_train, y_train)
    logisticRegr.predict(X_train[0].reshape(1, -1))
    logisticRegr.predict(X_train[0:10])
    predictions = logisticRegr.predict(y_test)
    score = logisticRegr.score(X_test, y_test)
    print(score)


# SVC
def svc(X_train, y_train, X_test, y_test):
    # create an SVC classifier model
    svclassifier = SVC(kernel='linear')
    # fit the model to train dataset
    svclassifier.fit(X_train, y_train)
    # make predictions using the trained model
    y_pred = svclassifier.predict(X_test)
    print(svclassifier.score(X_test, y_test))



# function make preprocessing combination
# input data and target
def makeCombination(x, y):
    listObj = ['Thickness', 'U_Size', 'U_Shape', 'Adhersion', 'S_Size', 'Bare', 'Bland', 'Normal', 'Mitoses']
    encoder = [lblEncoding(listObj, x), ordEncoding(listObj, x)]
    nameEnc = ['Label encoder', 'Ordinal encoder']
    scaler = [preprocessing.StandardScaler(), preprocessing.RobustScaler(), preprocessing.MaxAbsScaler(),
              preprocessing.MinMaxScaler()]
    nameSc = ['Standard scaler', 'Robust scaler', 'MaxAbs scaler', 'MinMax scaler']
    listDf = []
    listBestDf = []
    listClassifier = [DecisionTreeClassifier(criterion="entropy"), DecisionTreeClassifier(criterion="gini"),
                      LogisticRegression(), SVC()]

    # make each combination and store in listDf
    for i in range(len(encoder)):
        tempX = encoder[i]
        col = tempX.columns.values
        for j in range(len(scaler)):
            sc = scaler[j]
            tempX = sc.fit_transform(tempX)
            tempX = pd.DataFrame(tempX, columns=col)
            listDf.append(tempX)

    # search best encoder and scaler for each classifier
    for i in range(len(listClassifier)):
        classifer = listClassifier[i]
        scoreMax = 0
        indexMax = 0
        encBest = ''
        scBest = ''
        print(classifer)
        for j in range(len(listDf)):
            X_train, X_test, y_train, y_test = train_test_split(listDf[j], y, test_size=0.2)
            classifer.fit(X_train, y_train)
            score = classifer.score(X_test, y_test)
            print(score)
            if (scoreMax <= score):
                scoreMax = score
                indexMax = j
        listBestDf.append(listDf[indexMax])
        print('################Combiniation Result################')
        print('Best accuracy :', scoreMax)
        encBest = nameEnc[indexMax % 2]
        scBest = nameSc[indexMax % 4]
        print('Best combination : Encoding -> ', encBest, '  Scaling -> ', scBest, '\n')
        print('-----------------------------------------------------')

    # output list of best dataframe
    return listBestDf


# evaluation each model
def evaluation(x, y, classifier):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    score = cross_val_score(classifier, X_train, Y_train, cv=skf)
    print(classifier, '\nCross validation score :', score)
    classifier.fit(X_train, Y_train)
    print('Accuracy on test set :', classifier.score(X_test, Y_test))
    print('')


listClassifier = [DecisionTreeClassifier(criterion="entropy"), DecisionTreeClassifier(criterion="gini"),
                  LogisticRegression(), SVC()]

# listBestDf index 0=DecisionTree(entropy), 1=DecisionTree(gini) 2=LogisticRegression, 3=SVC
listBestDf = makeCombination(X, y)
for i in range(len(listBestDf)):
    evaluation(listBestDf[i], y, listClassifier[i])

# grid Decision Tree Entropy
X_train, X_test, y_train, y_test = train_test_split(listBestDf[0], y, test_size=0.2, shuffle=True, random_state=1)
param_grid = [{'max_features': np.arange(1, len(X_test.columns)), 'max_depth': np.arange(1, 20)}]
dt_entropy_gscv = GridSearchCV(listClassifier[0], param_grid, cv=2)
dt_entropy_gscv.fit(X_train, y_train)
print(dt_entropy_gscv.best_params_)
print('Best score :', dt_entropy_gscv.best_score_)

# grid Decision Tree Gini
X_train, X_test, y_train, y_test = train_test_split(listBestDf[1], y, test_size=0.2, shuffle=True, random_state=1)
param_grid = [{'max_features': np.arange(1, len(X_test.columns)), 'max_depth': np.arange(1, 10)}]
dt_gini_gscv = GridSearchCV(listClassifier[1], param_grid, cv=2, n_jobs=2)
dt_gini_gscv.fit(X_train, y_train)
print(dt_gini_gscv.best_params_)
print('Best score :', dt_gini_gscv.best_score_)

# grid Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(listBestDf[2], y, test_size=0.2, shuffle=True, random_state=1)
param_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
lr_gscv = GridSearchCV(listClassifier[2], param_grid, cv=2, n_jobs=2)
lr_gscv.fit(X_train, y_train)
print(lr_gscv.best_params_)
print('Best score :', lr_gscv.best_score_)

# grid SVC
X_train, X_test, y_train, y_test = train_test_split(listBestDf[2], y, test_size=0.2, shuffle=True, random_state=1)
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
svc_gscv = GridSearchCV(listClassifier[3], param_grid, cv=2, n_jobs=2)
svc_gscv.fit(X_train, y_train)
print(svc_gscv.best_params_)
print('Best score :', svc_gscv.best_score_)

print('\n---------After GridSearchCV---------\n')
dt_e = DecisionTreeClassifier(max_depth=5, max_features=3, criterion="entropy")
dt_g = DecisionTreeClassifier(max_depth=4, max_features=2, criterion="gini")
lr = LogisticRegression(C=100)
svc = SVC(C=1, gamma=1, kernel='rbf')

evaluation(listBestDf[0], y, dt_e)
evaluation(listBestDf[1], y, dt_g)
evaluation(listBestDf[2], y, lr)
evaluation(listBestDf[3], y, svc)
