# -*- coding: utf-8 -*-

from src.utils.get_data import load_data
from src.utils.get_data import get_datasets
from src.utils.get_data import concatenate_datasets

import os
import pandas as pd
import numpy as np
import unittest


class TestDataCollection(unittest.TestCase):
    def setUp(self):
        self.df = load_data(os.path.join('src', 'data', 'tweets.csv'))
        self.df2, self.df3 = get_datasets(os.path.join('src', 'data', 'labeled_data.csv'),
                                os.path.join('src', 'data',
                                             'hatespeech_text_label_vote_RESTRICTED_100K.csv'))
        self.df_concatenated = concatenate_datasets(os.path.join('data', 'tweets.csv'),
                                           self.df2,
                                           self.df3)
        
    def test_load_data(self):
        """ Test Case for correct loading of dataframes """
        self.assertIsInstance(load_data(os.path.join('src', 'data', 'tweets.csv')),
                              pd.core.frame.DataFrame)
        
        self.assertIsInstance(get_datasets(os.path.join('src', 'data', 'labeled_data.csv'),
                                           os.path.join('src', 'data',
                                                        'hatespeech_text_label_vote_RESTRICTED_100K.csv'))[0],
                                pd.core.frame.DataFrame)
        
        self.assertIsInstance(get_datasets(os.path.join('data', 'labeled_data.csv'),
                                           os.path.join('data',
                                                        'hatespeech_text_label_vote_RESTRICTED_100K.csv'))[1],
                                pd.core.frame.DataFrame)
        
        self.assertIsInstance(concatenate_datasets(os.path.join('src', 'data', 'tweets.csv'),
                                                   self.df2, self.df3),
                              pd.core.frame.DataFrame)
      
        
    def test_hatespeech_labels(self):
        """ Test Case for correct labeling into hatespeech and no hatespeech """
        # test if hatespeech labels are all 0 or 1
        self.assertTrue(np.all(np.logical_or(self.df_concatenated['hate_speech'] == 0,
                                             self.df_concatenated['hate_speech'] == 1)))


    def test_columns(self):
        """ Test Case for correct concatenation and labeling columns """
        # after preprocessing another column called "preprocessed"?
        self.assertTrue(np.all(self.df_concatenated.columns == ['label',
                                                                'text',
                                                                'hate_speech']))
    
if __name__ == "__main__":
    unittest.main()
