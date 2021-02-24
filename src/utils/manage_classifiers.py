import pickle
import os
from src.utils import constant
from src.classifiers import svm_classifier, logistic_regression, random_forest_classifier, decision_tree_classifier
from src.classifiers.ensemble_classifier import EnsembleClassifier


def save_classifier(model, vec):
    """
    Saves classifier to subfolder models in current working directory, folder models 
    needs to exist already

    Parameters
    ----------
    model:		        Model of implemented Classifiers
            			Trained Model 
    vec:        	    sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data
    """

    model_path = "./models/model.pkl"
    vec_path = "./models/vec.pkl"

    pickle.dump(model, open(model_path, "wb"))
    pickle.dump(vec, open(vec_path, "wb"))


def load_classifier(model_path, vec_path):
    """
    Loads classifier from subfolder .
    
    Parameters
    ----------
    model_path:	        String
                        The path where the classifier is stored
    vec_path:	        String
                        The path where the vectorizer is stored

    Returns
    ----------
    model:		        Model of implemented Classifiers
            			Trained Model
    vec:        	    sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data
    """

    model = pickle.load(open(model_path, "rb"))
    vec = pickle.load(open(vec_path, "rb"))

    return model, vec


def choose_and_create_classifier(classifier, x_data, y_data, df, method, ngrams):
    """
        Creates classifier on the given training data and parameters.

        Parameters
        ----------
        classifier:	        String
                            svm, decison_tree,random_forest, log_reg or ensemble

        x_data:	            Array
                            Features to train classifier

        y_data              Array
                            Labels to train classifier

        df                  Pandas dataframe
                            The dataframe containing the testing data for the classifier

        method: 		    String
                            Can be either "count" or "tfidf" for specifying method of feature weighting

        ngrams:             tuple (min_n, max_n), with min_n, max_n integer values
                            range for ngrams used for vectorization

        Returns
        ----------
        model:		        Model of implemented Classifiers
                			Trained Model
        vec:        	    sklearn CountVectorizer or TfidfVectorizer
                        	CountVectorizer or TfidfVectorizer fit and transformed for training data
        """
    # Choose classifier and create
    if classifier is constant.SVM:
        model, vec, x_test = svm_classifier.setup_svm_classifier(x_data, y_data, df[['preprocessed']],
                                                                 method=method, ngrams=ngrams)

    elif classifier is constant.DECISION_TREE:
        model, vec, x_test = decision_tree_classifier.setup_decision_tree_classifier(
            x_data, y_data, df[['preprocessed']], method=method, ngrams=ngrams)

    elif classifier is constant.RANDOM_FOREST:
        model, vec, x_test = random_forest_classifier.setup_random_forest_classifier(x_data,
                                                                                     y_data, df[['preprocessed']],
                                                                                     method=method, ngrams=ngrams)

    elif classifier is constant.LOGISTIC_REGRESSION:
        model, vec, x_test = logistic_regression.setup_log_reg_classifier(x_data,
                                                                          y_data, df[['preprocessed']], method=method,
                                                                          ngrams=ngrams)

    else:
        model = EnsembleClassifier()
        x_test, vec = model.train(x_data, y_data, df[['preprocessed']], method=method, ngrams=ngrams)

    return model, vec, x_test
