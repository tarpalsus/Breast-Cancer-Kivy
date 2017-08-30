# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import grid_search
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle

class MedicalPredictor(object):
    def __init__(self, path):
        self.classifier = None
        self.ax = None
        self.fig = None
        self.X, self.y = self.read_data(path)


    def plot_decision_boundary(self, model):
      padding = 0.6
      resolution = 0.05
      colors = ['royalblue', 'forestgreen', 'ghostwhite']
      # Calculate the boundaries
      x_min, x_max = self.X_train[:, 0].min(), self.X_train[:, 0].max()
      y_min, y_max = self.X_train[:, 1].min(), self.X_train[:, 1].max()
      x_range = x_max - x_min
      y_range = y_max - y_min
      x_min -= x_range * padding
      y_min -= y_range * padding
      x_max += x_range * padding
      y_max += y_range * padding


      xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                           np.arange(y_min, y_max, resolution))


      Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
      Z = Z.reshape(xx.shape)
    #

      cs = plt.contourf(xx, yy, Z, cmap=plt.cm.terrain)
    #

      for label in range(len(np.unique(self.y_train))):
        indices = np.where(self.y_train == label)
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train)
    #
      p = model.get_params()
      plt.axis('tight')

    def read_data(self, path):
        """Data munging"""
        df = pd.read_csv(path, sep=',', header=None, index_col=0)
        df[6] = pd.to_numeric(df[6], errors='coerce')
        df = df.dropna()
        y = df[10]
        X = df.drop([10], axis=1)
        return X, y

    def PCA(self, record, fig=None):
        """PCA and 2D visualisation"""
        pca = PCA(n_components=2)

        self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(self.X, self.y, test_size=0.25,
                                                                               random_state=3)

        lb = LabelBinarizer()
        self.y_train = np.array([number[0] for number in lb.fit_transform(self.y_train)])
        self.y_test = np.array([number[0] for number in lb.fit_transform(self.y_test)])

        pca.fit(self.X_train)
        # X_train2=np.copy(X_train)
        # X_test2=np.copy(X_test)
        self.X_train = pca.transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        self.record = record.drop([9], axis=1)
        self.record= pca.transform(self.record)
        T = pca.transform(self.X)
        plt.scatter(T[:, 0], T[:, 1], c=self.y)
        return fig


    def classify(self, model_name):
        """Load pre-trained classifiers"""
        if model_name == 'KNN':
            model = KNeighborsClassifier()
            parameters = {'n_neighbors': list(range(1, 10)), 'weights': ('uniform', 'distance')}
            with open('knn.pkl', 'rb') as f:
                classifier = pickle.load(f)
        if model_name == 'SVC':
            model = svm.SVC()
            parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 5, 10], 'class_weight': [{0: 20}]}
            with open('svc.pkl', 'rb') as f:
                classifier = pickle.load(f)
        if model_name == 'Random Forest':
            model = RandomForestClassifier()
            parameters = {'n_estimators': list(range(1, 10)), 'criterion': ('gini', 'entropy'),'class_weight': [{0: 20}]}
            with open('rf.pkl', 'rb') as f:
                classifier = pickle.load(f)
        classifier = grid_search.GridSearchCV(model, parameters, scoring='f1')

        classifier.fit(self.X_train, self.y_train)
        #with open('knn.pkl', 'wb') as f:
            #pickle.dump(classifier, f)
        pred = classifier.predict(self.X_test)
        predRecord = classifier.predict(self.record)
        score = metrics.accuracy_score(self.y_test, pred)
        print(score, classifier.best_estimator_)
        print(predRecord, self.y_test[0])
        self.plot_decision_boundary(classifier)
        plt.scatter(self.record[:, 0], self.record[:, 1], color='red')
        target_names = ['Benign', 'Malignant']
        class_report = classification_report(self.y_test, pred, target_names=target_names)

        return predRecord, score, class_report

    def main(self):
        '''Testing'''
        model = KNeighborsClassifier()
        parameters = {'n_neighbors': list(range(1, 10)), 'weights':('uniform', 'distance')}
        classifier, pred1 = self.classify(model)
        print(metrics.confusion_matrix(self.y_test, pred1))
        with open('knn.pkl', 'wb') as f:
            pickle.dump(model, f)
        self.plot_decision_boundary(classifier)
        plt.show()
        model = svm.SVC()
        parameters = {'kernel':('linear', 'rbf'), 'C': [1, 5, 10],'class_weight': [{0: 4}]}
        classifier, pred2 = self.classify(model, parameters)
        print(metrics.confusion_matrix(self.y_test,pred2))
        self.plot_decision_boundary(classifier)
        plt.show()
        with open('svc.pkl', 'wb') as f:
            pickle.dump(model, f)
        model = RandomForestClassifier()

        parameters= {'n_estimators':list(range(1,10)),'criterion':('gini','entropy')}
        classifier, pred3 =self.classify(model, parameters)
        print(metrics.confusion_matrix(self.y_test, pred3))
        self.plot_decision_boundary(classifier)

        plt.show()
        with open('rf.pkl', 'wb') as f:
            pickle.dump(model, f)
        model=LogisticRegression()
        parameters={'penalty':['l1','l2'],'C':[1, 5, 10],'class_weight':[{0:5}]}
        classifier, pred4 = self.classify(model, parameters)
        print(metrics.confusion_matrix(self.y_test, pred4))
        self.plot_decision_boundary(classifier)

        plt.show()
        model = svm.SVC()
        estimators = [('reduce_dim', PCA()), ('clf', svm.SVC())]
        pipe = Pipeline(estimators)
        parameters = {'clf__kernel':('linear', 'rbf'), 'clf__C':[1, 5, 10],'clf__class_weight':[{0:4}],
                      'reduce_dim__n_components':list(range(2,9))}
        classifier, pred2 = self.classify(pipe, parameters)
        print(metrics.confusion_matrix(self.y_test, pred2))
