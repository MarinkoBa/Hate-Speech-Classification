import svm


def fit_svm_classifier(training_data):
    """
    Train the svm classifier using TFIDF features.

    Parameters
    ----------
    training_data:  	    Pandas dataframe
                    	    The dataframe containing the training data for the SVM classifier

    Returns
    -------
    model:		            sklearn SVM Model
    			            Trained SVM Model

    vec:        	        sklearn CountVectorizer or TfidfVectorizer
                    	    CountVectorizer or TfidfVectorizer fit and transformed for training data

    training_data:  	    Pandas dataframe
                    	    The dataframe containing the training data for the SVM classifier
    """

    y_training = training_data["hate_speech"].values

    # TODO use the same method as in logistic regression in order to get features tfidf
    #  -> extract method from logistic regression to the new class e.g. "feature_builder"?

    vec = ...
    x_training = ...
    x_testing = ...

    SVM = svm.SVC()  # TODO investigate meaningful SVC params
    model = SVM.fit(x_training, y_training)

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
