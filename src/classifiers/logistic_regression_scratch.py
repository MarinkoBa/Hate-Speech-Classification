import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # move this to data?
from scipy import sparse
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#TODO clean docstring and remove prints

class LogisticRegression_scratch:
    """
    Class for fitting a LogsiticRegression Model
    """
    def __init__(self, lr=0.1, iterations=10000, use_intercept=True):
        """
        Init function for class

        Parameters
        ----------
        lr:		    Float
                           learning rate for model
        
        iterations:        integer
                           number of itertaions used for fitting
        
        use_intercept:     boolean
                           specifies if a intercept should be added to decesion function
    
        """          

        self.lr=lr
        self.iterations=iterations
        self.use_intercept=use_intercept
    
    def __add_intercept(self,X):
        """
        Add intercept for fitting

        Parameters
        ----------
        X:                 sparse matrix
                    	    Data used for Training
    
                    

        Returns
        -------
        data_intercept	    sparse matrix
                           Data with added intercept
        """
        
        intercept=sparse.csr_matrix(np.ones((X.shape[0],1)))
        return hstack((intercept, X))
        
    
    def __logistic_func(self,z):
        """
        Define logistic function necessary for logisticRegression

        Parameters
        ----------
        z:                 numeric
                    	    Should be dot product of feature vector and weights
    
                    

        Returns
        -------
        gz		    numpy array
                           result of logistic function
        """
        z = np.clip( z, -500, 500 )
        gz = 1 / (1 +np.exp(-z))
    
        return gz
        
    def __z(theta, X):
        """
        Define z with given weights and feature vector

        Parameters
        ----------
        theta:             numpy array
    			    weights
        X		    sparse matrix
    			    feature vector
    			    
        Returns
        -------
       
        z:                 numpy array
                    	    Dot product of feature vector and weights
                    	
        """
        z = X.dot(theta)
       
        return z
       
    def __log_likelihood(x, y, weights):
    
        """
        calculate log likelihood

        Parameters
        ----------
        X		    sparse matrix
    			    feature vector
        weights:           numpy array
    			    weights, same as theta
    			    
        Returns
        -------
       
        ll:                numeric
                    	    log likelihood
                    	
        """
        z = x.dot(weights)
        ll = np.sum( y*z - np.log(1 + np.exp(z)) )
        return ll
        
    def __gradient(self,X, h, y):
    
        """
        calculate gradient (dervative of likelihood)

        Parameters
        ----------
        X:      	    sparse matrix
    			    feature vector
        h:                 numeric
                           result of logistic function
        y:                 sparse matrix
                           training data
    			   
    			    
        Returns
        -------
       
        g:                 sparse matrix
                    	    gradient
                    	
        """
        g = X.T.dot(y - h)
        return g
        
    def __update_weights(self,weight, lr, gradient):
        """
        update weight using learning rate and gradient

        Parameters
        ----------
        weight:            numpy array
    			    weights, same as theta
        lr:		    Float
                           learning rate for model
        g:                 sparse matrix
                    	    gradient for logistic regression                           
    			    
        Returns
        -------
       
        updated_weight:    numpy array
                    	    updated weight 
                    	
        """
        updated_weight=weight + lr * gradient
        
        return updated_weight
        
    def fit(self, X, y):
        """
        fit model to data

        Parameters
        ----------
        X:      	    sparse matrix
    			    feature vector
        y:                 sparse matrix
                           training data                      

        """
        # intercept added if use_intercept true
        if self.use_intercept:
            X = self.__add_intercept(X)
            print("here")
            
        
        #initialising of theta
        
        self.theta = np.zeros(X.shape[1])
        
        #weight is updated 
        
        for i in range(self.iterations):
            z = X.dot(self.theta)
            h = self.__logistic_func(z)
            gradient = self.__gradient(X, h, y)
            self.theta = self.__update_weights(self.theta, self.lr,gradient)
            print(i)
            
    def predict_prob(self, X):
        """
        predict probabilties using logistic function

        Parameters
        ----------
        X:      	    sparse matrix
    			    feature vector              
        Returns
        -------
       
        probabilites:      numpy array
                    	    result of logistic function 
        """    
    
        if self.use_intercept:
            X = self.__add_intercept(X)
            
        probabilities= self.__logistic_func(X.dot(self.theta))
        
        return probabilities
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold
            
           
       
     
     
     
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
    vectorizer=CountVectorizer() #welche Parameter?
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
    tfidf_vectorizer=TfidfVectorizer() #welche Parameter?
    tfidf_vectorizer.fit_transform(training_data[columns].values)
    
    #build matrixes for training_features and testing_features
    training_features=tfidf_vectorizer.transform(training_data[columns].values)
    testing_features=tfidf_vectorizer.transform(testing_data[columns].values)
    
    
    
    return tfidf_vectorizer, training_features, testing_features
    
    
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
    
    model = LogisticRegression_scratch(lr=0.1, iterations=100,use_intercept=True)
    model.fit(x_training,y_training)
    
    return model.predict(x_testing),vec, x_testing
    			

                       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


