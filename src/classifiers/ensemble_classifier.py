import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .define_features import define_features_vectorizer
from .define_features import define_features_tfidf

import numpy as np


class EnsembleClassifier:
    """
    Ensemble classifier with training data in form of tf-idf or as term counts.
    Classifier predicts based on a SVM-, RandomForest- and LogisticRegression-Classifier due to a majority vote
    over the results of each of these classifiers.
    """

    def __init__(self):
        self.SVM = svm.SVC(kernel='linear', C=1, gamma=1)
        self.forest_model = RandomForestClassifier(random_state=0)
        self.logistic_model = log_reg_classifier = LogisticRegression(max_iter=1000, class_weight="balanced")

    def train(self, training_data, training_target, testing_data, features="preprocessed", method="count",
              ngrams=(1, 1)):
        """
        Fits the ensemble classifier with training data in form of tf-idf or as term counts.

        Parameters
        ----------
        training_data:  	Pandas dataframe
                        	The dataframe containing the training data for the classifier.
        training_target:    Pandas dataframe
                            The dataframe containing the training labels.
        testing_data:   	Pandas dataframe
                        	The dataframe containing the testing data for the classifier.
        features:           String or list of strings
                            Names of columns of df that are used for training the classifier.
        method: 		    String
                			Can be either "count" or "tfidf" for specifying method of
                            feature weighting.
        ngrams:            tuple (min_n, max_n), with min_n, max_n integer values
                           range for ngrams used for vectorization

        Returns
        -------

        X_testing:          Pandas dataframe
                            The dataframe containing the testing data in vectorized
                            form.
        """
        if method == "count":
            vec, X_training, X_testing = define_features_vectorizer(features, training_data, testing_data,
                                                                    ngramrange=ngrams)
        elif method == "tfidf":
            vec, X_training, X_testing = define_features_tfidf(features, training_data, testing_data, ngramrange=ngrams)
        else:
            print("Method has to be either count or tfidf")
            return 1

        self.SVM.fit(X_training, training_target.values.ravel())
        self.forest_model.fit(X_training, training_target.values.ravel())
        self.logistic_model.fit(X_training, training_target.values.ravel())

        return X_testing

    def predict(self, X_testing):
        """
            Predict the labels of X_testing using the trained Ensemble Classifier.
            In detail: Execute majority vote over the different used classifiers to predict result.
            Parameters
            ----------

            X_testing:   	        Pandas dataframe
                            	    The dataframe containing the testing data in vectorized
                                    form.

            Returns
            -------
            predictions:            Binary array
                                    The predictions array, containing 0 for no hate speech,
                                    1 for hate speech
            """
        y_pred_svm = self.SVM.predict(X_testing)
        y_pred_forest = self.forest_model.predict(X_testing)
        y_pred_logistic = self.forest_model.predict(X_testing)

        predicted_labels = np.vstack([y_pred_svm, y_pred_forest, y_pred_logistic])
        majority_vote = np.apply_along_axis(np.bincount, axis=0, arr=predicted_labels,
                                            minlength=np.max(predicted_labels) + 1)
        majority_vote = np.argmax(majority_vote, axis=0)

        return majority_vote
