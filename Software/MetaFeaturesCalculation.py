import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict, deque
import scipy.stats
from scipy.linalg import LinAlgError
import scipy.sparse
import sklearn.tree
import sklearn.neighbors
import sklearn.discriminant_analysis
import sklearn.naive_bayes
import sklearn.decomposition
import time

class metafeature_calculate:
    def __init__(self, X, y):
        print('#######################################')
        self.Missing = X.isnull()  #~np.isfinite(X)

        self.nominal = 0
        for i in range(X.shape[1]):
            if X.iloc[:, i].dtypes == 'object':
                self.nominal += 1
        self.numerical = X.shape[1] - self.nominal

        t0 = time.process_time()
        labels = 1 if len(y.shape) == 1 else y.shape[1]
        if labels == 1:
            y = y.values.reshape((-1, 1))
        self.all_occurence_dict = {}
        for i in range(labels):
            occurence_dict = defaultdict(float)
            for value in y[:, i]:
                occurence_dict[value] += 1
            self.all_occurence_dict[i] = occurence_dict
        #print('The time for calculating all_occurence_dict is {}'.format(
          #  time.process_time() - t0))
        y = pd.core.series.Series(y.reshape(1, -1)[0])
       # t1 = time.process_time()
        self.kurts = []
        for i in range(X.shape[1]):
            if X.iloc[:, i].dtypes != 'object':
                self.kurts.append(scipy.stats.kurtosis(X.iloc[:, i]))
        #print('The time for calculating kurtosis is {}'.format(
           # time.process_time() - t1))

        #def Skewness(self, X, y):
       # t2 = time.process_time()
        self.skews = []
        for i in range(X.shape[1]):
            if X.iloc[:, i].dtypes != 'object':
                self.skews.append(scipy.stats.skew(X.iloc[:, i]))
        #print(
         #   'The time for calculating skew is {}'.format(time.process_time() -
          #                                               t2))
        #return skews
        self.col = list(X.columns)
        self.symbols_per_column = []
        for j in self.col:
            if X[j].dtypes == 'object' or X[j].dtypes == 'O':
                b = X[j].unique()
                for i in range(len(b)):
                    X[j].loc[X[j] == b[i]] = i
                X[j]=X[j].astype("int")
                self.symbols_per_column.append(i + 1)
       # t4 = time.process_time()
        if y.dtypes == 'object' or y.dtypes == 'O':
            b = y.unique()
            for i in range(len(b)):
                y.loc[y == b[i]] = i 
            y=y.astype("int")   
        y=y.astype("int")
        ## PCA
        self.pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        for i in range(10):
            try:
                rs.shuffle(indices)
                self.pca.fit(X.iloc[indices])
#                 return pca
            except LinAlgError:
                pass
       # print('The time for calculating pca is {}'.format(time.process_time() -
       #                                                   t4))
        #return None
        self.newdataset_metafeas = []
       # t5 = time.process_time()
        self.newdataset_metafeas.append(self.ClassEntropy(X, y))
        self.newdataset_metafeas.append(self.ClassProbabilityMax(X, y))
        self.newdataset_metafeas.append(self.ClassProbabilityMean(X, y))
        self.newdataset_metafeas.append(self.ClassProbabilityMin(X, y))
        self.newdataset_metafeas.append(self.ClassProbabilitySTD(X, y))
       # print('The time for calculating metafeature_classes is {}'.format(
       #     time.process_time() - t5))
       # print(self.newdataset_metafeas)
       # t6 = time.process_time()
        self.newdataset_metafeas.append(self.DatasetRatio(X, y))
        self.newdataset_metafeas.append(self.InverseDatasetRatio(X, y))
        self.newdataset_metafeas.append(self.KurtosisMax(X, y))
        self.newdataset_metafeas.append(self.KurtosisMean(X, y))
        self.newdataset_metafeas.append(self.KurtosisMin(X, y))
        self.newdataset_metafeas.append(self.KurtosisSTD(X, y))
       # print('The time for calculating metafeature_Kurtosis is {}'.format(
       #     time.process_time() - t6))
       # print(self.newdataset_metafeas)
       # t7 = time.process_time()
        self.newdataset_metafeas.append(self.Landmark1NN(X, y))
        self.newdataset_metafeas.append(self.LandmarkDecisionNodeLearner(X, y))
        self.newdataset_metafeas.append(self.LandmarkDecisionTree(X, y))
        self.newdataset_metafeas.append(self.LandmarkLDA(X, y))
        self.newdataset_metafeas.append(self.LandmarkNaiveBayes(X, y))
        self.newdataset_metafeas.append(self.LandmarkRandomNodeLearner(X, y))
        #print('The time for calculating metafeature_Landmark is {}'.format(
        #    time.process_time() - t7))
        #print(self.newdataset_metafeas)
        #t8 = time.process_time()
        self.newdataset_metafeas.append(self.LogDatasetRatio(X, y))
        self.newdataset_metafeas.append(self.LogInverseDatasetRatio(X, y))
        self.newdataset_metafeas.append(self.LogNumberOfFeatures(X, y))
        self.newdataset_metafeas.append(self.LogNumberOfInstances(X, y))
        #print('The time for calculating metafeature_logs is {}'.format(
        #    time.process_time() - t8))
        #print(self.newdataset_metafeas)
        #t9 = time.process_time()
        self.newdataset_metafeas.append(self.NumberOfCategoricalFeatures(X, y))
        self.newdataset_metafeas.append(self.NumberOfClasses(X, y))
        self.newdataset_metafeas.append(self.NumberOfFeatures(X, y))
        self.newdataset_metafeas.append(
            self.NumberOfFeaturesWithMissingValues(X, y))
        self.newdataset_metafeas.append(self.NumberOfInstances(X, y))
        self.newdataset_metafeas.append(
            self.NumberOfInstancesWithMissingValues(X, y))
        self.newdataset_metafeas.append(self.NumberOfMissingValues(X, y))
        self.newdataset_metafeas.append(self.NumberOfNumericFeatures(X, y))
      #  print('The time for calculating metafeature_simples is {}'.format(
      #      time.process_time() - t9))
      #  print(self.newdataset_metafeas)
      #  t10 = time.process_time()
        self.newdataset_metafeas.append(
            self.PCAFractionOfComponentsFor95PercentVariance(X, y))
        self.newdataset_metafeas.append(self.PCAKurtosisFirstPC(X, y))
        self.newdataset_metafeas.append(self.PCASkewnessFirstPC(X, y))
        self.newdataset_metafeas.append(
            self.PercentageOfFeaturesWithMissingValues(X, y))
        self.newdataset_metafeas.append(
            self.PercentageOfInstancesWithMissingValues(X, y))
        self.newdataset_metafeas.append(self.PercentageOfMissingValues(X, y))
        self.newdataset_metafeas.append(self.RatioNominalToNumerical(X, y))
        self.newdataset_metafeas.append(self.RatioNumericalToNominal(X, y))
      #  print('The time for calculating metafeature_pcas is {}'.format(
      #      time.process_time() - t10))
      #  print(self.newdataset_metafeas)
       # t11 = time.process_time()
        self.newdataset_metafeas.append(self.SkewnessMax(X, y))
        self.newdataset_metafeas.append(self.SkewnessMean(X, y))
        self.newdataset_metafeas.append(self.SkewnessMin(X, y))
        self.newdataset_metafeas.append(self.SkewnessSTD(X, y))
        self.newdataset_metafeas.append(self.SymbolsMax(X, y))
        self.newdataset_metafeas.append(self.SymbolsMean(X, y))
        self.newdataset_metafeas.append(self.SymbolsMin(X, y))
        self.newdataset_metafeas.append(self.SymbolsSTD(X, y))
        self.newdataset_metafeas.append(self.SymbolsSum(X, y))
       # print('The time for calculating metafeature_Symbols is {}'.format(
       #     time.process_time() - t11))
        print('The metafeatures for the dataset are {}'.format(
            self.newdataset_metafeas))
        print('#######################################')
    def ClassEntropy(self, X, y):
        # def _calculate(self, X, y, categorical):
        all__occurence_dict = self.all_occurence_dict
        entropies = []
        for occurence_dict in all__occurence_dict.values():
            entropies.append(
                scipy.stats.entropy(
                    [occurence_dict[key] for key in occurence_dict], base=2))
        return np.mean(entropies)

    def ClassProbabilityMax(self, X, y):
        #def _calculate(self, X, y, categorical):
        occurences = self.all_occurence_dict
        max_value = -1
        if len(y.shape) == 2:
            for i in range(y.shape[1]):
                max_value = max(max_value, max(occurences[i].values()))
        else:
            max_value = max(occurences[0].values())
        return max_value / float(y.shape[0])

    def ClassProbabilityMean(self, X, y):
        occurence_dict = self.all_occurence_dict
        if len(y.shape) == 2:
            occurences = []
            for i in range(y.shape[1]):
                occurences.extend(
                    [occurrence for occurrence in occurence_dict[i].values()])
            occurences = np.array(occurences)
        else:
            occurences = np.array(
                [occurrence for occurrence in occurence_dict[0].values()],
                dtype=np.float64)
        return (occurences / y.shape[0]).mean()

    def ClassProbabilityMin(self, X, y):
        #def _calculate(self, X, y, categorical):
        occurences = self.all_occurence_dict
        min_value = y.shape[0]
        if len(y.shape) == 2:
            for i in range(y.shape[1]):
                min_value = min(min_value, min(occurences[i].values()))
        else:
            min_value = min(occurences[0].values())
        return min_value / float(y.shape[0])

    def ClassProbabilitySTD(self, X, y):
        #def _calculate(self, X, y, categorical):
        occurence_dict = self.all_occurence_dict

        if len(y.shape) == 2:
            stds = []
            for i in range(y.shape[1]):
                std = np.array(
                    [occurrence for occurrence in occurence_dict[i].values()],
                    dtype=np.float64)
                std = (std / y.shape[0]).std()
                stds.append(std)
            return np.mean(stds)
        else:
            occurences = np.array(
                [occurrence for occurrence in occurence_dict[0].values()],
                dtype=np.float64)
            return (occurences / y.shape[0]).std()

    def DatasetRatio(self, X, y):
        return float(X.shape[1]) / float(X.shape[0])

    def InverseDatasetRatio(self, X, y):
        return float(X.shape[0]) / float(X.shape[1])

    def KurtosisMax(self, X, y):
        Kurtosis = self.kurts
        maximum = np.nanmax(Kurtosis) if len(Kurtosis) > 0 else 0
        return maximum if np.isfinite(maximum) else 0

    def KurtosisMean(self, X, y):
        Kurtosis = self.kurts
        mean = np.nanmean(Kurtosis) if len(Kurtosis) > 0 else 0
        return mean if np.isfinite(mean) else 0

    def KurtosisMin(self, X, y):
        Kurtosis = self.kurts
        minimum = np.nanmin(Kurtosis) if len(Kurtosis) > 0 else 0
        return minimum if np.isfinite(minimum) else 0

    def KurtosisSTD(self, X, y):
        Kurtosis = self.kurts
        std = np.nanstd(Kurtosis) if len(Kurtosis) > 0 else 0
        return std if np.isfinite(std) else 0

    def Landmark1NN(self, X, y):

        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = 0.
        for train, test in kf.split(X, y):
            kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            if len(y.shape) == 1 or y.shape[1] == 1:
                kNN.fit(X.iloc[train], y.iloc[train])
            else:
                kNN = OneVsRestClassifier(kNN)
                kNN.fit(X.iloc[train], y.iloc[train])
            predictions = kNN.predict(X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions,
                                                       y.iloc[test])
        return accuracy / 5

    def LandmarkDecisionNodeLearner(self, X, y):

        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = 0.
        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy",
                max_depth=1,
                random_state=random_state,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=None)
            if len(y.shape) == 1 or y.shape[1] == 1:
                node.fit(X.iloc[train], y.iloc[train])
            else:
                node = OneVsRestClassifier(node)
                node.fit(X.iloc[train], y.iloc[train])
            predictions = node.predict(X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions,
                                                       y.iloc[test])
        return accuracy / 5

    def LandmarkDecisionTree(self, X, y):
        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = 0.
        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            tree = sklearn.tree.DecisionTreeClassifier(
                random_state=random_state)

            if len(y.shape) == 1 or y.shape[1] == 1:
                tree.fit(X.iloc[train], y.iloc[train])
            else:
                tree = OneVsRestClassifier(tree)
                tree.fit(X.iloc[train], y.iloc[train])

            predictions = tree.predict(X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions,
                                                       y.iloc[test])
        return accuracy / 5

    def LandmarkLDA(self, X, y):
        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = 0.
        try:
            for train, test in kf.split(X, y):
                lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
                )
                if len(y.shape) == 1 or y.shape[1] == 1:
                    lda.fit(X.iloc[train], y.iloc[train])
                else:
                    lda = OneVsRestClassifier(lda)
                    lda.fit(X.iloc[train], y.iloc[train])

                predictions = lda.predict(X.iloc[test])
                accuracy += sklearn.metrics.accuracy_score(
                    predictions, y.iloc[test])
            return accuracy / 5
        except scipy.linalg.LinAlgError as e:
            #self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
            return np.NaN
        except ValueError as e:
            #self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
            return np.NaN

    def LandmarkNaiveBayes(self, X, y):
        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = 0.
        for train, test in kf.split(X, y):
            nb = sklearn.naive_bayes.GaussianNB()
            if len(y.shape) == 1 or y.shape[1] == 1:
                nb.fit(X.iloc[train], y.iloc[train])
            else:
                nb = OneVsRestClassifier(nb)
                nb.fit(X.iloc[train], y.iloc[train])
            predictions = nb.predict(X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions,
                                                       y.iloc[test])
        return accuracy / 5

    def LandmarkRandomNodeLearner(self, X, y):

        kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        accuracy = 0.
        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy",
                max_depth=1,
                random_state=random_state,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=1)
            node.fit(X.iloc[train], y.iloc[train])
            predictions = node.predict(X.iloc[test])
            accuracy += sklearn.metrics.accuracy_score(predictions,
                                                       y.iloc[test])
        return accuracy / 5

    def LogDatasetRatio(self, X, y):
        return np.log(self.DatasetRatio(X, y))

    def LogInverseDatasetRatio(self, X, y):
        return np.log(self.InverseDatasetRatio(X, y))

    def LogNumberOfFeatures(self, X, y):
        return np.log(X.shape[1])

    def LogNumberOfInstances(self, X, y):
        return np.log(X.shape[0])

    def NumberOfCategoricalFeatures(self, X, y):
        return self.nominal

    def NumberOfClasses(self, X, y):
        res = []
        for i in self.all_occurence_dict.values():
            res.extend(i.keys())
        return len(set(res))

    def NumberOfFeatures(self, X, y):
        return X.shape[1]

    def NumberOfFeaturesWithMissingValues(self, X, y):
        missing = self.Missing
        num_missing = missing.sum(axis=0)
        return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

    def NumberOfInstances(self, X, y):
        return X.shape[0]

    def NumberOfInstancesWithMissingValues(self, X, y):
        missing = self.Missing
        num_missing = missing.sum(axis=1)
        return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

    def NumberOfMissingValues(self, X, y):
        missing = self.Missing
        num_missing = missing.sum(axis=1)
        return float(np.sum(num_missing))

    def NumberOfNumericFeatures(self, X, y):
        return self.numerical

    def PCAFractionOfComponentsFor95PercentVariance(self, X, y):
        pca_ = self.pca
        if pca_ is None:
            return np.NaN
        sum_ = 0.
        idx = 0
        while sum_ < 0.95 and idx < len(pca_.explained_variance_ratio_):
            sum_ += pca_.explained_variance_ratio_[idx]
            idx += 1
        return float(idx) / float(X.shape[1])

    def PCAKurtosisFirstPC(self, X, y):
        pca_ = self.pca
        if pca_ is None:
            return np.NaN
        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(X)
        pca_.components_ = components
        kurtosis = scipy.stats.kurtosis(transformed)
        return kurtosis[0]

    def PCASkewnessFirstPC(self, X, y):
        pca_ = self.pca
        if pca_ is None:
            return np.NaN
        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(X)
        pca_.components_ = components
        skewness = scipy.stats.skew(transformed)
        return skewness[0]

    def PercentageOfFeaturesWithMissingValues(self, X, y):
        return self.NumberOfFeaturesWithMissingValues(X, y) / X.shape[1]

    def PercentageOfInstancesWithMissingValues(self, X, y):
        return self.NumberOfInstancesWithMissingValues(X, y) / X.shape[0]

    def PercentageOfMissingValues(self, X, y):
        return self.NumberOfMissingValues(X, y) / (X.shape[0] * X.shape[1])

    def RatioNominalToNumerical(self, X, y):
        if self.numerical == 0:
            return 0.
        else:
            return self.nominal / self.numerical

    def RatioNumericalToNominal(self, X, y):
        if self.nominal == 0:
            return 0.
        else:
            return self.numerical / self.nominal

    def SkewnessMax(self, X, y):
        skews = self.skews
        maximum = np.nanmax(skews) if len(skews) > 0 else 0
        return maximum if np.isfinite(maximum) else 0

    def SkewnessMean(self, X, y):
        skews = self.skews
        mean = np.nanmean(skews) if len(skews) > 0 else 0
        return mean if np.isfinite(mean) else 0

    def SkewnessMin(self, X, y):
        skews = self.skews
        minimum = np.nanmin(skews) if len(skews) > 0 else 0
        return minimum if np.isfinite(minimum) else 0

    def SkewnessSTD(self, X, y):
        skews = self.skews
        std = np.nanstd(skews) if len(skews) > 0 else 0
        return std if np.isfinite(std) else 0

    def SymbolsMax(self, X, y):
        values = self.symbols_per_column
        if len(values) == 0:
            return 0
        return max(max(values), 0)

    def SymbolsMean(self, X, y):
        values = [val for val in self.symbols_per_column if val > 0]
        mean = np.nanmean(values)
        return mean if np.isfinite(mean) else 0

    def SymbolsMin(self, X, y):
        help_ = [i for i in self.symbols_per_column if i > 0]
        return min(help_) if help_ else 0

    def SymbolsSTD(self, X, y):
        values = [val for val in self.symbols_per_column if val > 0]
        std = np.nanstd(values)
        return std if np.isfinite(std) else 0

    def SymbolsSum(self, X, y):
        return np.nansum(self.symbols_per_column) if np.isfinite(
            np.nansum(self.symbols_per_column)) else 0