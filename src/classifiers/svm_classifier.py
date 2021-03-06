import sklearn.svm as svm
from .define_features import define_features_tfidf
from .define_features import define_features_vectorizer


def setup_svm_classifier(training_data, y_training, testing_data, features="preprocessed", method="count",ngrams=(1,1)):
    """
    Setup svm model using sklearn implementation

    Parameters
    ----------
    training_data:  	Pandas dataframe
                    	The dataframe containing the training data for the classifier
    y_training:   	    Pandas dataframe
                    	The dataframe containing the y training data for the classifier
    testing_data:   	Pandas dataframe
                    	The dataframe containing the testing data for the classifier
    features:         	String or list of strings if using multiple features
                    	Names of columns of df that are used for trainig the classifier
    method: 		    String
            			Can be either "count" or "tfidf" for specifying method of feature weighting
    ngrams:             tuple (min_n, max_n), with min_n, max_n integer values
                        range for ngrams used for vectorization


    Returns
    -------
    model:		        sklearn SVM Classifier
            			Trained SVM Classifier
    vec:        	    sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data
    """

    if method=="count":
        vec, x_training, x_testing = define_features_vectorizer(features, training_data, testing_data,ngramrange=ngrams)
    elif method=="tfidf":
        vec, x_training, x_testing = define_features_tfidf(features, training_data, testing_data,ngramrange=ngrams)
    else:
        print("Method has to be either count or tfidf")
        return 1

    SVM = svm.SVC(kernel='linear', C=1, gamma=1)
    model = SVM.fit(x_training, y_training.values.ravel())

    return model, vec, x_testing


def predict(model, x_testing):
    """
    Predict the labels of x_testing using SVM model.

    Parameters
    ----------
    model:             	    sklearn SVM Model
    			            Trained SVM Model

    x_testing:   	        Pandas dataframe
                    	    The dataframe containing the testing data for the SVM classifier

    Returns
    -------
    predictions:		    binary array
    			            predictions array, 0 for non hate speech, 1 for hate speech

    """

    predictions = model.predict(x_testing)

    return predictions
