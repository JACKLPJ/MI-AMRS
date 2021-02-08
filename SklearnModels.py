#class sklearn_models(time_per_run):   
import timeout_decorator
time_per_model=360
#print('****************************',time_per_model)
@timeout_decorator.timeout(time_per_model, use_signals=False)
def polynomial(X, y, preprocessing_dic):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(
        degree=preprocessing_dic[1]['degree'],
        interaction_only=preprocessing_dic[1]['interaction_only'],
        include_bias=preprocessing_dic[1]['include_bias'])
    return poly.fit_transform(X)

@timeout_decorator.timeout(time_per_model, use_signals=False)
def PCA(X, y, preprocessing_dic):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=preprocessing_dic[1]['n_components'],
              whiten=preprocessing_dic[1]['whiten'])
    pca.fit(X)
    return pca.fit_transform(X)

@timeout_decorator.timeout(time_per_model, use_signals=False)
def LDA(X, y, model_dic):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score, train_test_split
    #         X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                             y,
    #                                                             test_size=0.2)
    shrinkage = None if model_dic[1]['shrinkage'] < 0 else model_dic[1][
        'shrinkage']
    n_components = model_dic[1]['n_components']
    tol = model_dic[1]['tol']
    if 'shrinkage_factor' in model_dic[1]:
        shrinkage_factor = model_dic[1]['shrinkage_factor']
        clf = LinearDiscriminantAnalysis(shrinkage=shrinkage,
                                         n_components=n_components,
                                         tol=tol,
                                         shrinkage_factor=shrinkage_factor)
    else:
        clf = LinearDiscriminantAnalysis(shrinkage=shrinkage,
                                         n_components=n_components,
                                         tol=tol)
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25, random_state=42)

    #         accu = np.mean(cross_val_score(clf, X, y, cv=5))
    #         print("prediction accuracy: {}".format(accu))
    #         return accu
    return 'lda',clf,clf.fit(X_train, y_train).score(X_test, y_test)

@timeout_decorator.timeout(time_per_model, use_signals=False)
def QDA(X, y, model_dic):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score, train_test_split
    #         X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                             y,
    #                                                             test_size=0.2)

    clf = QuadraticDiscriminantAnalysis(
        reg_param=model_dic[1]['reg_param'])
    #         clf.fit(X_train, y_train)
    #         y_pred = clf.predict(X_test)
    #         accu = np.mean(cross_val_score(clf, X, y, cv=5))
    #         print("prediction accuracy: {}".format(accu))
    #         return accu
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25, random_state=42)

    #         accu = np.mean(cross_val_score(clf, X, y, cv=5))
    #         print("prediction accuracy: {}".format(accu))
    #         return accu
    return 'qda',clf,clf.fit(X_train, y_train).score(X_test, y_test)

@timeout_decorator.timeout(time_per_model, use_signals=False)
def SVM(X, y, model_dic):
    from sklearn import svm
    from sklearn.model_selection import cross_val_score, train_test_split
    #         X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                             y,
    #                                                             test_size=0.2)
    kernel = model_dic[1]['kernel']
    C = model_dic[1]['C']
    max_iter = model_dic[1]['max_iter']
    tol = model_dic[1]['tol']
    shrinking = True if model_dic[1]['shrinking'] else False
    gamma = model_dic[1]['gamma']
    f, k = 0, 0
    if 'degree' in model_dic[1]:
        degree = model_dic[1]['degree']
        f = 1
    if 'coef0' in model_dic[1]:
        coef0 = model_dic[1]['coef0']
        k = 1
    if f == 0 and k == 0:
        clf = svm.SVC(kernel=kernel,
                      C=C,
                      max_iter=max_iter,
                      tol=tol,
                      shrinking=shrinking,
                      gamma=gamma)
    elif f == 0 and k == 1:
        clf = svm.SVC(kernel=kernel,
                      C=C,
                      max_iter=max_iter,
                      tol=tol,
                      shrinking=shrinking,
                      gamma=gamma,
                      coef0=coef0)
    elif f == 1 and k == 1:
        clf = svm.SVC(kernel=kernel,
                      C=C,
                      max_iter=max_iter,
                      tol=tol,
                      shrinking=shrinking,
                      gamma=gamma,
                      coef0=coef0,
                      degree=degree)
    else:
        clf = svm.SVC(kernel=kernel,
                      C=C,
                      max_iter=max_iter,
                      tol=tol,
                      shrinking=shrinking,
                      gamma=gamma,
                      degree=degree)

#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         accu = np.mean(cross_val_score(clf, X, y, cv=5))
#         print("prediction accuracy: {}".format(accu))
#         return accu
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25, random_state=42)

    #         accu = np.mean(cross_val_score(clf, X, y, cv=5))
    #         print("prediction accuracy: {}".format(accu))
    #         return accu
    return 'svm',clf,clf.fit(X_train, y_train).score(X_test, y_test)

@timeout_decorator.timeout(time_per_model, use_signals=False)
def KNN(X, y, model_dic):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    #         X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                             y,
    #                                                             test_size=0.2)

    clf = KNeighborsClassifier(p=model_dic[1]['p'],
                               weights=model_dic[1]['weights'],
                               n_neighbors=model_dic[1]['n_neighbors'])
    #         neigh.fit(X_train, y_train)
    #         y_pred = neigh.predict(X_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25, random_state=42)

    #         accu = np.mean(cross_val_score(clf, X, y, cv=5))
    #         print("prediction accuracy: {}".format(accu))
    #         return accu
    return 'knn',clf,clf.fit(X_train, y_train).score(X_test, y_test)

@timeout_decorator.timeout(time_per_model, use_signals=False)
def MultinomialNB(X, y, model_dic):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import cross_val_score, train_test_split
    #         X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                             y,
    #                                                             test_size=0.2)

    clf = MultinomialNB(alpha=model_dic[1]['alpha'],
                        fit_prior=model_dic[1]['fit_prior'])
    #         clf.fit(X_train, y_train)
    #         y_pred = clf.predict(X_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25, random_state=42)

    #         accu = np.mean(cross_val_score(clf, X, y, cv=5))
    #         print("prediction accuracy: {}".format(accu))
    #         return accu
    return 'mnb',clf,clf.fit(X_train, y_train).score(X_test, y_test)

@timeout_decorator.timeout(time_per_model, use_signals=False)
def BernoulliNB(X, y, model_dic):
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.model_selection import cross_val_score, train_test_split
    #         X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                             y,
    #                                                             test_size=0.2)

    clf = BernoulliNB(alpha=model_dic[1]['alpha'],
                      fit_prior=model_dic[1]['fit_prior'])
    #         clf.fit(X_train, y_train)
    #         y_pred = clf.predict(X_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25, random_state=42)

    #         accu = np.mean(cross_val_score(clf, X, y, cv=5))
    #         print("prediction accuracy: {}".format(accu))
    #         return accu
    return 'bnb',clf,clf.fit(X_train, y_train).score(X_test, y_test)

@timeout_decorator.timeout(time_per_model, use_signals=False)
def DecisionTree(X, y, model_dic):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    if model_dic[1]['max_features'] > 0 and X.shape[1] < model_dic[1][
            'max_features']:
        max_features_ = X.shape[1]
    elif model_dic[1]['max_features'] <= 0:
        max_features_ = None
    else:
        max_features_ = int(model_dic[1]['max_features']) + 1
    clf = DecisionTreeClassifier(
        splitter=model_dic[1]['splitter'],
        min_samples_leaf=model_dic[1]['min_samples_leaf'],
        max_features=max_features_,  #int(model_dic[1]['max_features']) +
        #1 if model_dic[1]['max_features'] > 0 else None,
        min_weight_fraction_leaf=model_dic[1]['min_weight_fraction_leaf'],
        criterion=model_dic[1]['criterion'],
        min_samples_split=model_dic[1]['min_samples_split'],
        max_depth=int(model_dic[1]['max_depth']) +
        1 if model_dic[1]['max_depth'] > 0 else None,
        max_leaf_nodes=model_dic[1]['max_leaf_nodes']
        if model_dic[1]['max_leaf_nodes'] > 0 else None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25, random_state=42)

    #         accu = np.mean(cross_val_score(clf, X, y, cv=5))
    #         print("prediction accuracy: {}".format(accu))
    #         return accu
    return 'dt',clf,clf.fit(X_train, y_train).score(X_test, y_test)

@timeout_decorator.timeout(time_per_model, use_signals=False)
def RandomForest(X, y, model_dic):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    if model_dic[1]['max_features'] > 0 and X.shape[1] < model_dic[1][
            'max_features']:
        max_features_ = X.shape[1]
    elif model_dic[1]['max_features'] <= 0:
        max_features_ = None
    else:
        max_features_ = int(model_dic[1]['max_features']) + 1
    clf = RandomForestClassifier(
        bootstrap=model_dic[1]['bootstrap'],
        n_estimators=model_dic[1]['n_estimators'],
        min_samples_leaf=model_dic[1]['min_samples_leaf'],
        max_features=max_features_,  #int(model_dic[1]['max_features']) +
        # 1 if model_dic[1]['max_features'] > 0 else None,
        min_weight_fraction_leaf=model_dic[1]['min_weight_fraction_leaf'],
        criterion=model_dic[1]['criterion'],
        min_samples_split=model_dic[1]['min_samples_split'],
        max_depth=int(model_dic[1]['max_depth']) +
        1 if model_dic[1]['max_depth'] > 0 else None,
        max_leaf_nodes=model_dic[1]['max_leaf_nodes']
        if model_dic[1]['max_leaf_nodes'] > 0 else None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25, random_state=42)

    #         accu = np.mean(cross_val_score(clf, X, y, cv=5))
    #         print("prediction accuracy: {}".format(accu))
    #         return accu
    return 'rf',clf,clf.fit(X_train, y_train).score(X_test, y_test)

@timeout_decorator.timeout(time_per_model, use_signals=False)
def GradientBoosting(X, y, model_dic):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    if model_dic[1]['max_features'] > 0 and X.shape[1] < model_dic[1][
            'max_features']:
        max_features_ = X.shape[1]
    elif model_dic[1]['max_features'] <= 0:
        max_features_ = None
    else:
        max_features_ = int(model_dic[1]['max_features']) + 1
    clf = GradientBoostingClassifier(
        subsample=model_dic[1]['subsample'],
        loss=model_dic[1]['loss'],
        n_estimators=model_dic[1]['n_estimators'],
        min_samples_leaf=model_dic[1]['min_samples_leaf'],
        max_features=max_features_,  #int(model_dic[1]['max_features']) +
        #1 if model_dic[1]['max_features'] > 0 else None,
        min_weight_fraction_leaf=model_dic[1]['min_weight_fraction_leaf'],
        learning_rate=model_dic[1]['learning_rate'],
        min_samples_split=model_dic[1]['min_samples_split'],
        max_depth=int(model_dic[1]['max_depth']) +
        1 if model_dic[1]['max_depth'] > 0 else None,
        max_leaf_nodes=model_dic[1]['max_leaf_nodes']
        if model_dic[1]['max_leaf_nodes'] > 0 else None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25, random_state=42)


    return 'gb',clf,clf.fit(X_train, y_train).score(X_test, y_test)        

@timeout_decorator.timeout(time_per_model, use_signals=False)
def ExtraTrees(X, y, model_dic):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    if model_dic[1]['max_features'] > 0 and X.shape[1] < model_dic[1][
            'max_features']:
        max_features_ = X.shape[1]
    elif model_dic[1]['max_features'] <= 0:
        max_features_ = None
    else:
        max_features_ = int(model_dic[1]['max_features']) + 1
    clf = ExtraTreesClassifier(
        bootstrap=model_dic[1]['bootstrap'],
        n_estimators=model_dic[1]['n_estimators'],
        min_samples_leaf=model_dic[1]['min_samples_leaf'],
        max_features=max_features_,  #int(model_dic[1]['max_features']) +
        #1 if model_dic[1]['max_features'] > 0 else None,
        min_weight_fraction_leaf=model_dic[1]['min_weight_fraction_leaf'],
        criterion=model_dic[1]['criterion'],
        min_samples_split=model_dic[1]['min_samples_split'],
        max_depth=int(model_dic[1]['max_depth']) +
        1 if model_dic[1]['max_depth'] > 0 else None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25, random_state=42)

    #         accu = np.mean(cross_val_score(clf, X, y, cv=5))
    #         print("prediction accuracy: {}".format(accu))
    #         return accu
    return 'etree',clf,clf.fit(X_train, y_train).score(X_test, y_test)

@timeout_decorator.timeout(time_per_model, use_signals=False)
def XGB(X, y, model_dic):
    from xgboost.sklearn import XGBClassifier
    from sklearn.model_selection import cross_val_score, train_test_split

    clf = XGBClassifier(
        colsample_bytree=model_dic[1]['colsample_bytree'],
        colsample_bylevel=model_dic[1]['colsample_bylevel'],
        alpha=model_dic[1]['alpha'],
        # scale_pos_weight=model_dic[1]['scale_pos_weight'],
        learning_rate=model_dic[1]['learning_rate'],
        max_delta_step=model_dic[1]['max_delta_step'],
        base_score=model_dic[1]['base_score'],
        n_estimators=model_dic[1]['n_estimators'],
        subsample=model_dic[1]['subsample'],
        reg_lambda=model_dic[1]['reg_lambda'],
        min_child_weight=model_dic[1]['min_child_weight'],
        max_depth=int(model_dic[1]['max_depth']) +
        1 if model_dic[1]['max_depth'] > 0 else None,
        gamma=model_dic[1]['gamma'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25, random_state=42)

    #         accu = np.mean(cross_val_score(clf, X, y, cv=5))
    #         print("prediction accuracy: {}".format(accu))
    #         return accu
    return 'xgb',clf,clf.fit(X_train, y_train).score(X_test, y_test)
