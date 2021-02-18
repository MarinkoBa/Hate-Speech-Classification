# -*- coding: utf-8 -*-

from src.utils.get_data import load_data
from src.utils.get_data import get_datasets
from src.utils.get_data import concatenate_datasets
from src.utils.dataset_balancer import balance_data

import os
import pandas as pd
import unittest


class TestDataBalancer(unittest.TestCase):
    def setUp(self):
        self.df = load_data(os.path.join(os.path.pardir, 'src', 'data', 'tweets.csv'))
        self.df2, self.df3 = get_datasets(os.path.join(os.path.pardir, 'src', 'data', 'labeled_data.csv'),
                                          os.path.join(os.path.pardir, 'src', 'data',
                                                       'hatespeech_text_label_vote_RESTRICTED_100K.csv'))
        self.df_concatenated = concatenate_datasets(os.path.join(os.path.pardir, 'src', 'data', 'tweets.csv'),
                                                    self.df2,
                                                    self.df3)


    def test_balance_data(self):
    
        x_balanced, y_balanced = balance_data(self.df_concatenated[['text']],
                                              self.df_concatenated[['hate_speech']])
    
        self.assertIsInstance(y_balanced,
                              pd.core.frame.DataFrame)

        self.assertIsInstance(x_balanced,
                              pd.core.frame.DataFrame)
    
        self.assertEquals(x_balanced.shape, y_balanced.shape)
    
    
    

if __name__ == "__main__":
    unittest.main()    
                              
