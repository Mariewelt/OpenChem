from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn import metrics

from data import get_fp, get_desc, normalize_desc, cross_validation_split

from mordred import Calculator, descriptors


class RandomForestQSAR(object):
    def __init__(self, model_type='classifier', feature_type='fingerprints', n_estimators=100, n_ensemble=5):
        super(RandomForestQSAR, self).__init__()
        self.n_estimators = n_estimators
        self.n_ensemble = n_ensemble
        self.model = []
        self.model_type = model_type
        if self.model_type == 'classifier':
            for i in range(n_ensemble):
                self.model.append(RFC(n_estimators=n_estimators))
        elif self.model_type == 'regressor':
            for i in range(n_ensemble):
                self.model.append(RFR(n_estimators=n_estimators))
        else:
            raise ValueError('invalid value for argument')
        self.feature_type = feature_type
        if self.feature_type == 'descriptors':
            self.calc = Calculator(descriptors, ignore_3D=True)
            self.desc_mean = [0] * self.n_ensemble

    def load_model(self, path):
        self.model = []
        for i in range(self.n_ensemble):
            m = joblib.load(path + str(i) + '.pkl')
            self.model.append(m)
        if self.feature_type == 'descriptors':
            arr = np.load(path + 'desc_mean.npy', 'rb')
            self.desc_mean = arr

    def save_model(self, path):
        assert self.n_ensemble == len(self.model)
        for i in range(self.n_ensemble):
            joblib.dump(self.model[i], path + str(i) + '.pkl')
        if self.feature_type == 'descriptors':
            np.save(path + 'desc_mean.npy', self.desc_mean)

    def fit_model(self, data):
        eval_metrics = []
        if self.feature_type == 'fingerprints':
            fps = get_fp(data.smiles)
        elif self.feature_type == 'descriptors':
            fps, _, _ = get_desc(data.smiles, self.calc)
        if self.model_type == 'classifier':
            cross_val_data, cross_val_labels = \
                cross_validation_split(fps, data.binary_labels)
        elif self.model_type == 'regressor':
            cross_val_data, cross_val_labels = \
                cross_validation_split(fps, data.property)
        for i in range(self.n_ensemble):
            train_sm = np.concatenate(cross_val_data[:i] + cross_val_data[(i + 1):])
            test_sm = cross_val_data[i]
            train_labels = np.concatenate(cross_val_labels[:i] + cross_val_labels[(i + 1):])
            test_labels = cross_val_labels[i]
            if self.feature_type == 'descriptors':
                train_sm, desc_mean = normalize_desc(train_sm)
                self.desc_mean[i] = desc_mean
                test_sm, _ = normalize_desc(test_sm, desc_mean)
            self.model[i].fit(train_sm, train_labels.ravel())
            predicted = self.model[i].predict(test_sm)
            if self.model_type == 'classifier':
                fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted)
                eval_metrics.append(metrics.auc(fpr, tpr))
                metrics_type = 'AUC'
            elif self.model_type == 'regressor':
                r2 = metrics.r2_score(test_labels, predicted)
                eval_metrics.append(r2)
                metrics_type = 'R^2 score'

        return eval_metrics, metrics_type

    def predict(self, smiles, average=True):
        if self.feature_type == 'fingerprints':
            fps = get_fp(smiles)
            assert len(smiles) == len(fps)
            clean_smiles = []
            clean_fps = []
            nan_smiles = []
            for i in range(len(fps)):
                if np.isnan(sum(fps[i])):
                    nan_smiles.append(smiles[i])
                else:
                    clean_smiles.append(smiles[i])
                    clean_fps.append(fps[i])
            clean_fps = np.array(clean_fps)
        elif self.feature_type == 'descriptors':
            clean_fps, clean_smiles, nan_smiles = get_desc(smiles, self.calc)
        prediction = []
        if len(clean_fps) > 0:
            for i in range(self.n_ensemble):
                m = self.model[i]
                if self.feature_type == 'descriptors':
                    clean_fps, _ = normalize_desc(clean_fps, self.desc_mean[i])
                prediction.append(m.predict(clean_fps))
            prediction = np.array(prediction)
            if average:
                prediction = prediction.mean(axis=0)
        assert len(clean_smiles) == len(prediction)

        return clean_smiles, prediction, nan_smiles


class SVMQSAR(object):
    def __init__(self, model_type='classifier', n_ensemble=5):
        super(SVMQSAR, self).__init__()
        self.n_ensemble = n_ensemble
        self.model = []
        self.model_type = model_type
        if self.model_type == 'classifier':
            for i in range(n_ensemble):
                self.model.append(SVC())
        elif self.model_type == 'regressor':
            for i in range(n_ensemble):
                self.model.append(SVR())
        else:
            raise ValueError('invalid value for argument')

    def load_model(self, path):
        self.model = []
        for i in range(self.n_ensemble):
            m = joblib.load(path + str(i) + '.pkl')
            self.model.append(m)

    def save_model(self, path):
        assert self.n_ensemble == len(self.model)
        for i in range(self.n_ensemble):
            joblib.dump(self.model[i], path + str(i) + '.pkl')

    def fit_model(self, data, cross_val_data, cross_val_labels):
        eval_metrics = []
        for i in range(self.n_ensemble):
            train_sm = np.concatenate(cross_val_data[:i] + cross_val_data[(i + 1):])
            test_sm = cross_val_data[i]
            train_labels = np.concatenate(cross_val_labels[:i] + cross_val_labels[(i + 1):])
            test_labels = cross_val_labels[i]
            fp_train = get_fp(train_sm)
            fp_test = get_fp(test_sm)
            self.model[i].fit(fp_train, train_labels.ravel())
            predicted = self.model[i].predict(fp_test)
            if self.model_type == 'classifier':
                fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted)
                eval_metrics.append(metrics.auc(fpr, tpr))
                metrics_type = 'AUC'
            elif self.model_type == 'regressor':
                r2 = metrics.r2_score(test_labels, predicted)
                eval_metrics.append(r2)
                metrics_type = 'R^2 score'
        return eval_metrics, metrics_type

    def predict(self, smiles, average=True):
        fps = get_fp(smiles)
        assert len(smiles) == len(fps)
        clean_smiles = []
        clean_fps = []
        nan_smiles = []
        for i in range(len(fps)):
            if np.isnan(sum(fps[i])):
                nan_smiles.append(smiles[i])
            else:
                clean_smiles.append(smiles[i])
                clean_fps.append(fps[i])
        clean_fps = np.array(clean_fps)
        prediction = []
        if len(clean_fps) > 0:
            for m in self.model:
                prediction.append(m.predict(clean_fps))
            prediction = np.array(prediction)
            if average:
                prediction = prediction.mean(axis=0)
        assert len(clean_smiles) == len(prediction)
        return clean_smiles, prediction, nan_smiles
