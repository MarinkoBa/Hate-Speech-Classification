import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .define_features import define_features_vectorizer
from .define_features import define_features_tfidf

import numpy as np


# TODO: 1. Documentation of the Classifier, 2. Maybe change style of implementation, 3. Evaluate Classfier
class EnsembleClassifier:
    def __init__(self):
        self.SVM = svm.SVC(kernel='linear', C=1, gamma=1)
        self.forest_model = RandomForestClassifier(random_state=0)
        self.logistic_model = log_reg_classifier = LogisticRegression(max_iter=1000, class_weight="balanced")

    def train(self, training_data, training_labels, test_labels, features="preprocessed", method="count"):

        if method == "count":
            vec, X_training, X_testing = define_features_vectorizer(features, training_data, test_labels)
        elif method == "tfidf":
            vec, X_training, X_testing = define_features_tfidf(features, training_data, test_labels)
        else:
            print("Method has to be either count or tfidf")
            return 1

        self.SVM.fit(X_training, training_labels.values.ravel())
        self.forest_model.fit(X_training, training_labels.values.ravel())
        self.logistic_model.fit(X_training, training_labels.values.ravel())

        return X_testing

    def predict(self, x):
        y_pred_svm = self.SVM.predict(x)
        y_pred_forest = self.forest_model.predict(x)
        y_pred_logistic = self.forest_model.predict(x)

        predicted_labels = np.vstack([y_pred_svm, y_pred_forest, y_pred_logistic])
        majority_vote = np.apply_along_axis(np.bincount, axis=0, arr=predicted_labels,
                                            minlength=np.max(predicted_labels) + 1)
        majority_vote = np.argmax(majority_vote, axis=0)

        return majority_vote
