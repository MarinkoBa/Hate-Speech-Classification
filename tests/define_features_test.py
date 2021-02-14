from src.utils.get_data import load_data
from src.utils.get_data import get_datasets
from src.utils.get_data import concatenate_datasets
from src.utils.get_data import split_data
from src.classifiers.define_features import define_features_vectorizer
from src.classifiers.define_features import define_features_tfidf


import os
import pandas as pd
import numpy as np
import unittest
import sklearn
import scipy


class TestDefineFeatures(unittest.TestCase):
    def setUp(self):
        self.df = load_data(os.path.join(os.path.pardir, 'src', 'data', 'tweets.csv'))
        self.df2, self.df3 = get_datasets(os.path.join(os.path.pardir, 'src', 'data', 'labeled_data.csv'),
                                os.path.join(os.path.pardir, 'src', 'data',
                                             'hatespeech_text_label_vote_RESTRICTED_100K.csv'))
        self.df_concatenated = concatenate_datasets(os.path.join(os.path.pardir, 'src', 'data', 'tweets.csv'),
                                           self.df2,
                                           self.df3)
        self.test_set = pd.DataFrame (['This is the first document.',
                                      'This document is the second document.',
                                      'And this is the third one.',
                                      'Is this the first document?'],
                                      columns=["text"])
        self.test_result_count = [[0, 1, 1, 1, 0, 0, 1, 0, 1],
                                  [0, 2, 0, 1, 0, 1, 1, 0, 1],
                                  [1, 0, 0, 1, 1, 0, 1, 1, 1],
                                  [0, 1, 1, 1, 0, 0, 1, 0, 1]]
                                  

        
    def test_define_features_vectorizer(self):
        """ Test Case for Feature Defintion using CountVectorizer"""
        
        training_data, testing_data, training_y, testing_y = split_data(self.df_concatenated,
                                                                    'text',
                                                                    'hate_speech',
                                                                    0.25)
        
        vectorizer, training_features, testing_features = define_features_vectorizer("text",
                                                                                      training_data, 
                                                                                      testing_data)
        vec2, training_features2, testing_features2 =define_features_vectorizer("text",
                                                                                 self.test_set)  
                                                                                      
        """ Test correct data types """                
                                                              
        self.assertIsInstance(vectorizer,
                              sklearn.feature_extraction.text.CountVectorizer)
        
        self.assertIsInstance(training_features, scipy.sparse.csr.csr_matrix)
        
        self.assertIsInstance(testing_features, scipy.sparse.csr.csr_matrix)
        
                                                                                              
        """ Test  None case """                
                                                            
        self.assertIsInstance(vec2,
                              sklearn.feature_extraction.text.CountVectorizer)
        
        self.assertIsInstance(training_features2, scipy.sparse.csr.csr_matrix)
        
        self.assertIsNone(testing_features2)
        
        
        self.assertTrue((training_features2.toarray()==self.test_result_count).all())
        
        
    def test_define_features_tfidf(self):
        """ Test Case for Feature Defintion using Tfidf"""
        
        training_data, testing_data, training_y, testing_y = split_data(self.df_concatenated,
                                                                    'text',
                                                                    'hate_speech',
                                                                    0.25)
        
        vectorizer, training_features, testing_features = define_features_tfidf("text",
                                                                                      training_data, 
                                                                                      testing_data)
        vec2, training_features2, testing_features2 =define_features_tfidf("text",
                                                                                 self.test_set)  
                                                                                      
        """ Test correct data types """                
                                                              
        self.assertIsInstance(vectorizer,
                              sklearn.feature_extraction.text.TfidfVectorizer)
        
        self.assertIsInstance(training_features, scipy.sparse.csr.csr_matrix)
        
        self.assertIsInstance(testing_features, scipy.sparse.csr.csr_matrix)
        
                                                                                              
        """ Test  None case """                
                                                            
        self.assertIsInstance(vec2,
                              sklearn.feature_extraction.text.TfidfVectorizer)
        
        self.assertIsInstance(training_features2, scipy.sparse.csr.csr_matrix)
        
        self.assertIsNone(testing_features2)

        
        
       
      

    
if __name__ == "__main__":
    unittest.main()
