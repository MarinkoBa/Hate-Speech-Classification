import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def define_features_vectorizer(columns, training_data,testing_data):
    """
    Define the features for classification using CountVectorizer.

    Parameters
    ----------
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
    
    
    
    
def define_features_tfidf(columns, training_data,testing_data):
    """
    Define the features for classification using TFIDF.

    Parameters
    ----------
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
    
