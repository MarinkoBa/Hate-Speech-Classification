# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from classifiers.define_features import define_features_vectorizer
from classifiers.define_features import define_features_tfidf
 
def setup_random_forest_classifier(training_data,
                                   training_target,
                                   testing_data,
                                   features = "preprocessed",
                                   method = "count"):
    """
    Set up the decision tree classifier with training data in form of tf-idf or as term counts.

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
                    
    Returns
    -------
    model:		        sklearn.ensemble.RandomForestClassifier
            			Trained RandomForestClassifier
    vec:        	    sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed
                        for training data
    X_testing:          Pandas dataframe
                        The dataframe containing the testing data in vectorized
                        form.
    """
    

    # generate x and y training data
    y_training = training_target
    
    if method == "count":
        vec, X_training, X_testing = define_features_vectorizer(features,
                                                                training_data,
                                                                testing_data)
    elif method == "tfidf":
        vec, X_training, X_testing = define_features_tfidf(features,
                                                           training_data,
                                                           testing_data)
    else:
        print("Method has to be either count or tfidf")
        return 1
    

    random_forest = RandomForestClassifier(random_state=0)
    model = random_forest.fit(X_training, y_training)
    
    return model, vec, X_testing    



def predict(model, X_testing):
    """
    Predict the labels of X_testing using the trained Random Forest Classifier.
    Parameters
    ----------
    model:             	    sklearn.ensemble.RandomForestClassifier
                			Trained RandomForestClassifier
    X_testing:   	        Pandas dataframe
                    	    The dataframe containing the testing data in vectorized
                            form.
    
    Returns
    -------
    predictions:            Binary array
                            The predictions array, containing 0 for no hate speech,
                            1 for hate speech
    """

    predictions = model.predict(X_testing)

    return predictions
