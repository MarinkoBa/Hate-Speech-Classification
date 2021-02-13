import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .define_features import define_features_vectorizer
from .define_features import define_features_tfidf

import numpy as np


class EnsembleClassifier:
    """
    Ensemble classifier with training data in form of tf-idf or as term counts.
    Classifier predicts based on a SVM-, RandomForest- and LogisticRegression-Classifier
    due to a majority vote over the results of each of these classifiers.
    """

    def __init__(self, svm_kernel='linear', svm_c=1, svm_gamma=1, forest_random_state=0,
                 logistic_max_iter=1000, logistic_class_weight="balanced"):
        """
        Initialize Ensemble Classifier.
        Parameters
        ----------
        svm_kernel:  	        String
                                Specifies the kernel type to be used in the algorithm.

        svm_c:                  Numeric
                                Regularization parameter. The strength of the regularization
                                is inversely proportional to C. Must be strictly positive.

        svm_gamma:   	        Numeric
                                Kernel coefficient for 'rbf', 'poly' and 'sigmoid'

        forest_random_state:    Integer
                                Controls both the randomness of the bootstrapping of the samples used when
                                building trees (if bootstrap=True) and the sampling of the features to consider
                                when looking for the best split at each node (if max_features < n_features).

        logistic_max_iter: 	    Integer
                                Maximum number of iterations taken for the solvers to converge.

        logistic_class_weight:  String, dict or ‘balanced’
                                default=None, Weights associated with classes in the form {class_label: weight}.
                                If not given, all classes are supposed to have weight one.
        """

        self.SVM = svm.SVC(kernel=svm_kernel, C=svm_c, gamma=svm_gamma)
        self.forest_model = RandomForestClassifier(random_state=forest_random_state)
        self.logistic_model = LogisticRegression(max_iter=logistic_max_iter, class_weight=logistic_class_weight)


    def train(self, training_data, training_target, testing_data, features="preprocessed",
              method="count", ngrams=(1, 1)):
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
        ngrams:             tuple (min_n, max_n), with min_n, max_n integer values
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
        y_pred_logistic = self.logistic_model.predict(X_testing)

        # stack predictions of the three classifiers
        predicted_labels = np.vstack([y_pred_svm, y_pred_forest, y_pred_logistic])

        # count how often each label appears
        majority_vote = np.apply_along_axis(np.bincount, axis=0, arr=predicted_labels,
                                            minlength=np.max(predicted_labels) + 1)
        # decide for the label which has the highest count
        majority_vote = np.argmax(majority_vote, axis=0)

        return majority_vote
