#!/usr/bin/env python
# coding: utf-8
# %%

# %%
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
import time
import timeout_decorator
import SklearnModels as sm
import DataModeling8w as dm
'''
X:数据集的特征
y:数据集的标签
k:数据集的邻居数 
N:每个邻居选择前N个best model
time_per_model:每个模型的运行限时
data_pre_processing:是否需要数据预处理 
'''


class Automl:
    #time_per_model=360

    def __init__(self, k=10, N=10, DoEnsembel=True, data_pre_processing=False, address_data_feats_featurized='/root/pmf-automl-master/data_feats_featurized.csv', address_pipeline='/root/pmf-automl-master/pipelines.json',address_Top50='/root/datasetTop50.csv'):
        self.data_pre_processing = data_pre_processing
        self.k = k
        self.N = N
        self.address_data_feats_featurized=address_data_feats_featurized
        self.address_pipeline=address_pipeline
        self.address_Top50=address_Top50
        dm.add_dff=address_data_feats_featurized
        dm.add_pipeline=address_pipeline
        dm.add_dT50=address_Top50
        self.DoEnsembel = DoEnsembel
        self.y = []
        #sm.time_per_model = time1
        self.ensemble_clf = []

    def pre_processing_X(self, X):
        col = list(X.columns)
        for j in col:
            if X[j].dtypes == 'object' or X[j].dtypes == 'O':
                b = X[j].unique()
                for i in range(len(b)):
                    X[j].loc[X[j] == b[i]] = i
                X[j] = X[j].astype("int")
        if self.data_pre_processing:
            t = time.perf_counter()
            if preprocessing_dics[i][0] == 'polynomial':
                X = sm.polynomial(X, y, preprocessing_dics[i])
            else:
                X = sm.PCA(X, y, preprocessing_dics[i])
            print('The runtime of preprocessing is {}.\n'.format(
                time.perf_counter() - t))
        return X

    def fit(self, Xtrain, ytrain):
        X = Xtrain.copy(deep=True)
        y = ytrain.copy(deep=True)
        self.y = ytrain.copy(deep=True)

        preprocessing_dics, model_dics = dm.data_modeling(
            X, y, self.k, self.N, self.address_data_feats_featurized, self.address_pipeline,self.address_Top50).result  #_preprocessor
        print('#######################################')
        n = len(preprocessing_dics)
        y = y.astype('int')
        accuracy = []
        great_models = []
        for i in range(n):
            if self.data_pre_processing:
                t = time.perf_counter()
                if preprocessing_dics[i][0] == 'polynomial':
                    X = sm.polynomial(X, y, preprocessing_dics[i])
                else:
                    X = sm.PCA(X, y, preprocessing_dics[i])

                print('The runtime of preprocessing is {}.\n'.format(
                    time.perf_counter() - t))

            t = time.perf_counter()
            if model_dics[i][0] == 'xgradient_boosting':
                try:
                    Str, Clf, acc = sm.XGB(X, y, model_dics[i])
                except:
                    acc = -1
            elif model_dics[i][0] == 'gradient_boosting':
                try:
                    Str, Clf, acc = sm.GradientBoosting(X, y, model_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'lda':
                try:
                    Str, Clf, acc = sm.LDA(X, y, model_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'extra_trees':
                try:
                    Str, Clf, acc = sm.ExtraTrees(X, y, model_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'random_forest':
                try:
                    Str, Clf, acc = sm.RandomForest(X, y, model_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'decision_tree':
                try:
                    Str, Clf, acc = sm.DecisionTree(X, y, model_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'libsvm_svc':
                try:
                    Str, Clf, acc = sm.SVM(X, y, model_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'k_nearest_neighbors':
                try:
                    Str, Clf, acc = sm.KNN(X, y, model_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'bernoulli_nb':
                try:
                    Str, Clf, acc = sm.BernoulliNB(X, y, model_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'multinomial_nb':
                try:
                    Str, Clf, acc = sm.MultinomialNB(X, y, model_dics[i])
                except:
                    acc = -1

            else:
                try:
                    Str, Clf, acc = sm.QDA(X, y, model_dics[i])
                except:
                    acc = -1
            if acc > -1:
                accuracy.append(acc)
                great_models.append(Clf)

            print('The runtime of model {} is {}, and the accuracy is {};\n'.
                  format(model_dics[i][0],
                         time.perf_counter() - t, acc))
            print('#######################################')

        sort_id = sorted(range(len(accuracy)),
                         key=lambda m: accuracy[m],
                         reverse=True)
        if self.DoEnsembel:
            mean_acc = np.mean(accuracy)
            estimators_stacking = [(str(sort_id[0]), great_models[sort_id[0]])]
            id_n = len(sort_id)
            id_i = 1
            base_acc_s = [accuracy[sort_id[0]]]
            while accuracy[sort_id[id_i]] >= mean_acc:
                estimators_stacking.append(
                    (str(sort_id[id_i]), great_models[sort_id[id_i]]))

                base_acc_s.append(accuracy[sort_id[id_i]])
                id_i += 1
            eclf_stacking = StackingClassifier(estimators=estimators_stacking)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42)
            accuracy.append(
                eclf_stacking.fit(X_train, y_train).score(X_test, y_test))

            self.ensemble_clf = eclf_stacking
            return eclf_stacking
        else:
            self.clf = great_models[sort_id[0]]
            #allresult = [great_models[sort_id[0]], accuracy[sort_id[0]]]
            return self.clf

    def predict(self, Xtest):
        X_Test = Xtest.copy(deep=True)
        X_Test = self.pre_processing_X(X_Test)
        if self.DoEnsembel:
            ypre = self.ensemble_clf.predict(X_Test)
        else:
            ypre = self.clf.predict(X_Test)
        if self.y.dtypes == 'object' or self.y.dtypes == 'O':
            b = self.y.unique()
            return [b[i] for i in ypre]
        return ypre
