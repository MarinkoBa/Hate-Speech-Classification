import pandas as pd
from sklearn.model_selection import train_test_split # move this to data?
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def define_features_vectorizer(df, columns, training_data,testing_data):
    """
    Define the features fro classification using CountVectorizer.

    Parameters
    ----------
    df:             	Pandas dataframe
                    	The dataframe containing both training_data and testing_data
    column:         	String
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
    vectorizer=CountVectorizer() #welche Parameter?
    vectorizer.fit_transform(training_data[columns].values)
    
    #build matrixes for training_features and testing_features
    training_features=vectorizer.transform(training_data[columns].values)
    testing_features=vectorizer.transform(testing_data[columns].values)
    
    
    
    return vectorizer, training_features, testing_features
    
    
    
    
def define_features_tfidf(df, columns, training_data,testing_data):
    """
    Define the features fro classification using CountVectorizer.

    Parameters
    ----------
    df:             	Pandas dataframe
                    	The dataframe containing both training_data and testing_data
    column:         	String
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
    #intialise Tfidfvectorizer and fit transform to data
    tfidf_vectorizer=TfidfVectorizer() #welche Parameter?
    tfidf_vectorizer.fit_transform(training_data[columns].values)
    
    #build matrixes for training_features and testing_features
    training_features=tfidf_vectorizer.transform(training_data[columns].values)
    testing_features=tfidf_vectorizer.transform(testing_data[columns].values)
    
    
    
    return vectorizer, training_features, testing_features
    
    
 



