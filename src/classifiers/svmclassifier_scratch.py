import numpy as np
from sklearn.utils import shuffle
from .define_features import define_features_tfidf
from .define_features import define_features_vectorizer


class SVMClassifier_scratch:
    """
    Class for fitting a SVM Model (scratch implementation) using SGD
    """
    def __init__(self, lr=1e-3, C=1.0, iterations=10):
        """
        Init function for class

        Parameters
        ----------
        lr:		           Float
                           learning rate for model

        C:                 Float
                           regularization parameter

        iterations:        integer
                           number of iterations used for fitting

        """

        self.lr = lr
        self.lam = C
        self.iterations = iterations

    def compute_gradients(self, x_i, y_i):
        """
        Computes gradients w.r.t. w and b

        Parameters
        ----------
        x_i:		       numpy array
                           the i-th x data point, for which the gradients are computed

        y_i:               Integer, -1 or 1
                           label of the i-th data point

        Returns
        -------
        dw:                 numpy array
                            the computed gradient, w.r.t. w

        db:                 Float
                            the computed gradient, w.r.t. b
        """
        dw = 0
        db = 0
        if y_i * (np.dot(x_i, self.w) - self.b) >= 1:  # if correct prediction, only margin updated
            dw = 2 * self.lam * self.w
            db = 0
        else:
            dw = 2 * self.lam * self.w - np.dot(x_i, y_i)  # if wrong prediction, margin and bias updated
            db = y_i

        return dw, db

    def update_gradients(self, dw, db):
        """
        Updates weights and bias (w and b) using the learning rate

        Parameters
        ----------
        dw:                 numpy array
                            the computed gradient, w.r.t. w

        db:                 Float
                            the computed gradient, w.r.t. b
        """
        self.w = self.w - self.lr * dw
        self.b = self.b - (self.lr * db)

    def fit(self, X, Y):
        """
        Fits model to data.

        Parameters
        ----------
        X:      	       csr_matrix NxD
                           feature matrix, training data tfidf or vectorizer

        Y:                 Series N
                           labels corresponding to X

        Returns
        -------
        w:                 numpy array
                           trained, final weights w

        b:                 Float
                           trained, final bias b
        """
        X = X.toarray()  # convert X to ndarray
        Y = Y.to_numpy()  # convert Y to numpy array
        Y[Y == 0] = -1  # convert all zeros to -1, the SVM works with -1 and 1 values

        self.w = np.zeros(X.shape[1])
        self.b = 0

        for iter in range(self.iterations):
            X, Y = shuffle(X, Y)
            for idx, x_i in enumerate(X):
                dw, db = self.compute_gradients(x_i, Y[idx])
                self.update_gradients(dw, db)


    def predict(self, X):
        """
        Predicts labels for unseen X data

        Parameters
        ----------
        X:      	       csr_matrix NxD
                           feature matrix, test data tfidf or vectorizer

        Returns
        -------
        y_pred:            numpy array
                           predictions for the test data X, contains -1 and 1
        """
        X = X.toarray()
        y_pred = np.sign(np.dot(X, self.w) - self.b)
        y_pred[y_pred == -1] = 0  # convert the values -1 back to 0
        return y_pred


def setup_svm_classifier(training_data, y_training, testing_data, features, method="count", ngrams=(1,1)):
    """
    Setup SVM classifier model using own implementation

    Parameters
    ----------
    training_data:  	Pandas dataframe
                    	The dataframe containing the training data for the classifier

    testing_data:   	Pandas dataframe
                    	The dataframe containing the testing data for the classifier

    y_training:   	    Pandas dataframe
                    	The dataframe containing the y training data for the classifier

    features:         	String or list of strings if using multiple features
                    	Names of columns of df that are used for trainig the classifier

    method: 		    String
    			        Can be either "count" or "tfidf" for specifying method of feature weighting
    ngrams:             tuple (min_n, max_n), with min_n, max_n integer values
                        range for ngrams used for vectorization


    Returns
    -------
    model:		        SVM Classifier (scratch implementation)
    			        Trained SVM Classifier from own implementation

    vec:        	    sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data

    x_testing:  	    Pandas dataframe
                    	The dataframe containing the test data for the SVM classifier
    """
    # generate x and y training data

    if method == "count":
        vec, x_training, x_testing = define_features_vectorizer(features, training_data, testing_data,ngramrange=ngrams)
    elif method == "tfidf":
        vec, x_training, x_testing = define_features_tfidf(features, training_data, testing_data,ngramrange=ngrams)
    else:
        print("Method has to be either count or tfidf")
        return 1

    # train classifier

    model = SVMClassifier_scratch()
    model.fit(x_training, y_training)

    return model, vec, x_testing


def predict(model, X_testing):
    """
    Predict the labels of X_testing using the trained svm model.
    Parameters
    ----------
    model:             SVM Model
                       Trained SVM Model
    X_testing:   	   Pandas dataframe
                       The dataframe containing the testing data in vectorized form.

    Returns
    -------
    predictions:       Binary array
    			       The predictions array, containing 0 for no hate speech,
                       1 for hate speech
    """
    predictions = model.predict(X_testing)

    return predictions
