from src.utils.get_data import load_data
from src.utils.get_data import get_datasets
from src.utils.get_data import concatenate_datasets
from src.utils.get_data import split_data
from src.classifiers.svm_classifier import setup_svm_classifier
from src.classifiers.svm_classifier import predict 


import os
import pandas as pd
import numpy as np
import unittest
import sklearn
import scipy


class TestSvmClassifier(unittest.TestCase):
    def setUp(self):
        self.df = load_data(os.path.join('src', 'data', 'tweets.csv'))
        self.df2, self.df3 = get_datasets(os.path.join('src', 'data', 'labeled_data.csv'),
                                os.path.join('src', 'data',
                                             'hatespeech_text_label_vote_RESTRICTED_100K.csv'))
        self.df_concatenated = concatenate_datasets(os.path.join('data', 'tweets.csv'),
                                           self.df2,
                                           self.df3)
               
        self.training_data, self.testing_data, self.training_y, self.testing_y = split_data(self.df_concatenated,
                                                                    'text',
                                                                    'hate_speech',
                                                                    0.25)

                                  

        
    def test_setup_svm_classifier(self):
        """ Test Case for setup of SVM Classifier"""
        

        model ,vec, x_testing=setup_svm_classifier(self.training_data, self.training_y, self.testing_data,"text", method="count")
        
        model2 ,vec_tfidf, x_testing2=setup_svm_classifier(self.training_data, self.training_y, self.testing_data,"text", method="tfidf")
  
                                                                                      
        """ Test correct data types for countVectorizer"""                
                                                              
        self.assertIsInstance(vec,
                              sklearn.feature_extraction.text.CountVectorizer)
        
        self.assertIsInstance(x_testing, scipy.sparse.csr.csr_matrix)
        
        self.assertIsInstance(model, sklearn.svm.SVC)
        
        """ Test correct data types TfidfVectorizer"""                
                                                              
        self.assertIsInstance(vec_tfidf,
                              sklearn.feature_extraction.text.TfidfVectorizer)
        
        self.assertIsInstance(x_testing2, scipy.sparse.csr.csr_matrix)
        
        self.assertIsInstance(model2, sklearn.svm.SVC)
        
        
        """ Test correct behaviour for wrong method"""  
        
        self.assertTrue(setup_svm_classifier(self.training_data, self.training_y, self.testing_data,"text", method="ijfsiohf"),
                        1)  
                        
    def test_predict(self):
        """ Test Case for Predict for Svm Classifier"""
        

        model ,vec, x_testing=setup_svm_classifier(self.training_data, self.training_y, self.testing_data,"text", method="count")
        
        model2 ,vec_tfidf, x_testing2=setup_svm_classifier(self.training_data, self.training_y, self.testing_data,"text", method="tfidf")
  
                                                                                      
        """ Test correct data types and corrrect range of predicted values (1,0) for predict with countVectorizer"""                
                                                              
        self.assertIsInstance(predict(model,x_testing),
                              np.ndarray)
        
        self.assertTrue(([0,1] ==np.unique(predict(model2,x_testing2))).all())

        
        """ Test correct data types and corrrect range of predicted values (1,0) for predict with tfidfVectorizer"""                
                                                              
        self.assertIsInstance(predict(model2,x_testing2),
                              np.ndarray)
                              
        self.assertTrue(([0,1] ==np.unique(predict(model2,x_testing2))).all())
        
 
        
                                                                                                         
                                                                                                                

if __name__ == "__main__":
    unittest.main()
