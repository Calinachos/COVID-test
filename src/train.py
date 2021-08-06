import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler


def train(df):
    """
        Function that trains and saves a model
        There are 4 model available to be used
        :param df: encoded input data frame
        :return:
    """
    X = df['train'].to_numpy()
    X2 = []
    for ab in X:
        ab = np.array(ab)
        X2.append(ab)
    X = X2
    Y = np.array(df['rezultat'])

    # over-sampling
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, Y)
    X = X_resampled
    Y = y_resampled
    print(len(Y))

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # 1. Logistic Regression
    reg_log = LogisticRegression()
    reg_log.fit(X_train, Y_train)
    # save model for later
    filename = 'finalized_model_LR.sav'
    pickle.dump(reg_log, open(filename, 'wb'))
    Y_pred = reg_log.predict(X_test)
    print("Metrics for Logistic Regression Model:")
    print(metrics.classification_report(Y_test, Y_pred))

    # 2. Random Forrest
    reg_rf = RandomForestClassifier()
    reg_rf.fit(X_train, Y_train)
    # save model for later
    filename = 'finalized_model_RF.sav'
    pickle.dump(reg_rf, open(filename, 'wb'))
    Y_pred = reg_rf.predict(X_test)
    print("Metrics for Random Forrest Model:")
    print(metrics.classification_report(Y_test, Y_pred))

    # 3. SVC
    reg_svc = SVC()
    reg_svc.fit(X_train, Y_train)
    # save model for later
    filename = 'finalized_model_SVC.sav'
    pickle.dump(reg_svc, open(filename, 'wb'))
    Y_pred = reg_svc.predict(X_test)
    print("Metrics for SVC Model:")
    print(metrics.classification_report(Y_test, Y_pred))

    # 4. KNN
    reg_knn = KNeighborsClassifier()
    reg_knn.fit(X_train, Y_train)
    # save model for later
    filename = 'finalized_model_KNN.sav'
    pickle.dump(reg_knn, open(filename, 'wb'))
    y_pred = reg_knn.predict(X_test)
    print("Metrics for K-Neighbors Classifier:")
    print(metrics.classification_report(Y_test, y_pred))


def demo_test(df):
    """
       :param df: preprocessed data frame, no splitting needed
       :return:
    """
    X = df['train'].to_numpy()
    X2 = []
    for ab in X:
        ab = np.array(ab)
        X2.append(ab)
    X = X2
    Y = np.array(df['rezultat'])
    # load the trained model - one of the 4 existing
    filename = 'finalized_model_LR.sav'
    filename1 = 'finalized_model_RF.sav'
    filename2 = 'finalized_model_SVC.sav'
    filename3 = 'finalized_model_KNN.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X, Y)
    print("LR accuracy = " + str(result))
    loaded_model = pickle.load(open(filename1, 'rb'))
    result = loaded_model.score(X, Y)
    print("RF accuracy = " + str(result))
    loaded_model = pickle.load(open(filename2, 'rb'))
    result = loaded_model.score(X, Y)
    print("SVC accuracy = " + str(result))
