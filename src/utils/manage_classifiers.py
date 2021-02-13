import pickle
import os

def save_classifier(model, vec):
    """
    Saves classifier to subfolder models in current working directory, folder models 
    needs to exist already

    Parameters
    ----------
    model:		        Model of implemented Classifiers
            			Trained Model 
    vec:        	    sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data
    """
    
    model_path="./models/model.pkl"
    vec_path="./models/vec.pkl"
    
    pickle.dump(model,open(model_path,"wb"))
    pickle.dump(vec,open(vec_path,"wb"))   
   
    
    
def load_classifier(model_path,vec_path):
    """
    Saves classifier to subfolder models in current working directory,
    folder models needs to exist already.
    
    Parameters
    ----------
    model_path:	        String
                        The path where the classifier is stored
    vec_path:	        String
                        The path where the vectorizer is stored

    Returns
    ----------
    model:		        Model of implemented Classifiers
            			Trained Model
    vec:        	    sklearn CountVectorizer or TfidfVectorizer
                    	CountVectorizer or TfidfVectorizer fit and transformed for training data
    """
    
    
    
    model=pickle.load(open(model_path,"rb"))
    vec=pickle.load(open(vec_path,"rb")) 
     
    return model, vec 
