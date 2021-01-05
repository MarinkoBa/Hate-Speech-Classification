import pandas as pd
from sklearn.model_selection import train_test_split # move this to data?
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import os


def define_features_vectorizer(df, columns, training_data,testing_data):
    """
    Define the features for classification using CountVectorizer.

    Parameters
    ----------
    df:             	Pandas dataframe
                    	The dataframe containing both training_data and testing_data
    column:         	String or list of strings if using multiple columns
                    	Names of columns of df that are used for trainig the classifier
    training_data:  	Pandas dataframe  
                    	The dataframe containing the training data for the classifier
    testing_data:   	Pandas dataframe  
                    	The dataframe containing the testing data for the classifier
                    

    Returns
    -------
    vectorizer:     	sklearn CountVectorizer
                    	CountVectorizer fit and transformed for training data
    training_features: sparse matrix
    			Document-term matrix for training data
    testing_features:  sparse matrix
    			Document-term matrix for testing data
    """
    #intialise Countvectorizer and fit transform to data
    vectorizer=CountVectorizer()  # TODO investigate meaningful params
    vectorizer.fit_transform(training_data[columns].values)
    
    #build matrixes for training_features and testing_features
    training_features=vectorizer.transform(training_data[columns].values)
    testing_features=vectorizer.transform(testing_data[columns].values)
    
    
    
    return vectorizer, training_features, testing_features
    
    
    
    
def define_features_tfidf(df, columns, training_data,testing_data):
    """
    Define the features for classification using TFIDF.

    Parameters
    ----------
    df:             	Pandas dataframe
                    	The dataframe containing both training_data and testing_data
    column:         	String or list of strings if using multiple columns
                    	Names of columns of df that are used for trainig the classifier
    training_data:  	Pandas dataframe  
                    	The dataframe containing the training data for the classifier
    testing_data:   	Pandas dataframe  
                    	The dataframe containing the testing data for the classifier
                    

    Returns
    -------
    tfidf_vectorizer:     	sklearn TfidfVectorizer
                    	TfidfVectorizer fit and transformed for training data
    training_features: sparse matrix
    			Document-term matrix for training data
    testing_features:  sparse matrix
    			Document-term matrix for testing data
    """
    #intialise Tfidfvectorizer and fit transform to data
    tfidf_vectorizer=TfidfVectorizer()  # TODO investigate meaningful params
    tfidf_vectorizer.fit_transform(training_data[columns].values)
    
    #build matrixes for training_features and testing_features
    training_features=tfidf_vectorizer.transform(training_data[columns].values)
    testing_features=tfidf_vectorizer.transform(testing_data[columns].values)
    
    
    
    return tfidf_vectorizer, training_features, testing_features
    
    
def setup_log_reg_classifier(df, training_data, testing_data, features, method="count"):
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
        vec, x_training, x_testing = define_features_vectorizer(df, features, training_data, testing_data)
    elif method=="tfidf":
        vec, x_training, x_testing = define_features_tfidf(df, features, training_data, testing_data)
    else:
        print("Method has to be either count or tfidf")
        return 1
    
    #train classifier
    
    log_reg_classifier=LogisticRegression(max_iter=1000,class_weight="balanced")
    model=log_reg_classifier.fit(x_training,y_training)
    
    return model,vec, x_testing
    
    
def save_classifier(model,vec):
    """
    Saves classifier to subfolder models in current working directory, folder models needs to exist already

    Parameters
    ----------
    model:		sklearn LogisticRegression Model
    			Trained LogistciRegression Model 
    vec:        	sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data
    """
    
    model_path="./models/model.pkl"
    vec_path="./models/vec.pkl"
    
    pickle.dump(model,open(model_path,"wb"))
    pickle.dump(vec,open(vec_path,"wb"))   
   
    
    
def load_classifier(model_path,vec_path):
    """
    Saves classifier to subfolder models in current working directory, folder models needs to exist already
    
    Parameters
    ----------
    model_path:	String
                       The path where the classifier is stored
    vec_path:	        String
                       The path where the vectorizer is stored

    Returns
    ----------
    model:		sklearn LogisticRegression Model
    			Trained LogistciRegression Model 
    vec:        	sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data
    """
    
    
    
    model=pickle.load(open(model_path,"rb"))
    vec=pickle.load(open(vec_path,"rb")) 
     
    return model, vec 
    
    

    	
	

	
    
    
 



