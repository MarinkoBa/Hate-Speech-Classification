import pandas as pd
from sklearn.model_selection import train_test_split # move this to data?
from sklearn.linear_model import LogisticRegression
from .define_features import define_features_vectorizer
from .define_features import define_features_tfidf
import pickle
import os



    
def setup_log_reg_classifier(training_data, testing_data, features, method="count"):
    """
    Define the features fro classification using TFIDF.

    Parameters
    ----------
    df:             	Pandas dataframe
                    	The dataframe containing both training_data and testing_data
    training_data:  	Pandas dataframe  
                    	The dataframe containing the training data for the classifier
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
    
    y_training=training_data["hate_speech"].values
    
    if method=="count":
        vec, x_training, x_testing = define_features_vectorizer(features, training_data, testing_data)
    elif method=="tfidf":
        vec, x_training, x_testing = define_features_tfidf(features, training_data, testing_data)
    else:
        print("Method has to be either count or tfidf")
        return 1
    
    #train classifier
    
    log_reg_classifier=LogisticRegression(max_iter=1000,class_weight="balanced")
    model=log_reg_classifier.fit(x_training,y_training)
    
    return model,vec, x_testing
    
    

    model=pickle.load(open(model_path,"rb"))
    vec=pickle.load(open(vec_path,"rb")) 
     
    return model, vec 
    
    

    	
	

	
    
    
 



