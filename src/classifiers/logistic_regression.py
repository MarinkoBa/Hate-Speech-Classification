import pandas as pd
from sklearn.linear_model import LogisticRegression
from .define_features import define_features_vectorizer
from .define_features import define_features_tfidf
import pickle
import os



    
def setup_log_reg_classifier(training_data, y_training, testing_data, features="preprocessed", method="count"):
    """
    Setup logistic regression model using sklearn implementation

    Parameters
    ----------
    df:             	Pandas dataframe
                    	The dataframe containing both training_data and testing_data
    training_data:  	Pandas dataframe  
                    	The dataframe containing the training data for the classifier
    y_training:   	    Pandas dataframe
                    	The dataframe containing the y training data for the classifier
    testing_data:   	Pandas dataframe  
                    	The dataframe containing the testing data for the classifier
    features:         	String or list of strings if using multiple features
                    	Names of columns of df that are used for trainig the classifier
    method: 		String
    			Can be either "count" or "tfidf" for specifying method of feature weighting
                    

    Returns
    -------
    model:		sklearn LogisticRegression Model
    			Trained LogistciRegression Model
    vec:        	sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data
    """
    

    #generate x and y training data
    
    #y_training=training_data["hate_speech"].values
    
    if method=="count":
        vec, x_training, x_testing = define_features_vectorizer(features, training_data, testing_data)
    elif method=="tfidf":
        vec, x_training, x_testing = define_features_tfidf(features, training_data, testing_data)
    else:
        print("Method has to be either count or tfidf")
        return 1
    
    #train classifier
    
    log_reg_classifier=LogisticRegression(max_iter=1000,class_weight="balanced")
    model=log_reg_classifier.fit(x_training,y_training.values.ravel())
    
    return model,vec, x_testing
    
def predict(model, X_testing):
    """
    Predict the labels of X_testing using the trained logistic regression model.
    Parameters
    ----------
    model:             sklearn.tree.DecisionTreeClassifier Model
                       Trained DecisionTreeClassifier Model
    X_testing:   	Pandas dataframe
                       The dataframe containing the testing data in vectorized form.
    
    Returns
    -------
    predictions:       Binary array
    			The predictions array, containing 0 for no hate speech,
                       1 for hate speech
    """

    predictions = model.predict(X_testing)

    return predictions
    
    

 
    

    	
	

	
    
    
 



