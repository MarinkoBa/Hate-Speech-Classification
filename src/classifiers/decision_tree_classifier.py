# -*- coding: utf-8 -*-
from sklearn.tree import DecisionTreeClassifier
from .define_features import define_features_vectorizer
from .define_features import define_features_tfidf
 
def setup_decision_tree_classifier(training_data,
                                   training_target,
                                   testing_data,
                                   features = "preprocessed",
                                   method = "count",
                                   ngrams=(1,1)):
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
    method: 		 String
                        Can be either "count" or "tfidf" for specifying method of 
                        feature weighting.
    ngrams:            tuple (min_n, max_n), with min_n, max_n integer values
                       range for ngrams used for vectorization
                    
    Returns
    -------
    model:		        sklearn.tree.DecisionTreeClassifier Model
            			Trained DecisionTreeClassifier Model
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
                                                                testing_data,ngramrange=ngrams)
    elif method == "tfidf":
        vec, X_training, X_testing = define_features_tfidf(features,
                                                           training_data,
                                                           testing_data,ngramrange=ngrams)
    else:
        print("Method has to be either count or tfidf")
        return 1
    
    ### Consider performing dimensionality reduction (PCA, ICA, or Feature selection) 
    # beforehand to give your tree a better chance of finding features that are 
    # discriminative.
    decision_tree = DecisionTreeClassifier(random_state=0)
    model = decision_tree.fit(X_training, y_training.values.ravel())
    
    return model, vec, X_testing    



def predict(model, X_testing):
    """
    Predict the labels of X_testing using the trained Decision Tree Classifier.
    Parameters
    ----------
    model:             	    sklearn.tree.DecisionTreeClassifier Model
                			Trained DecisionTreeClassifier Model
    X_testing:   	        Pandas dataframe
                    	    The dataframe containing the testing data in vectorized
                            form.
    
    Returns
    -------
    predictions:		    Binary array
    			            The predictions array, containing 0 for no hate speech,
                            1 for hate speech
    """

    predictions = model.predict(X_testing)

    return predictions
