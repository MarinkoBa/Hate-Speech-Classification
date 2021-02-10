from src.classifiers.ensemble_classifier import EnsembleClassifier
from src.utils.get_data import load_data
from src.utils.get_data import get_datasets
from src.utils.get_data import concatenate_datasets
from src.utils.get_data import split_data

import os
import pandas as pd
import numpy as np
import unittest
import scipy
import sklearn

# TODO: Check if tests works correctly
class TestEnsemble(unittest.TestCase):
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

        self.ensemble1 = EnsembleClassifier()
        self.ensemble2 = EnsembleClassifier()


    def test_train_ensemble(self):
        """ Test Case for Feature Defintion using CountVectorizer"""

        # count
        x_testing = self.ensemble1.train(self.training_data, self.training_y, self.testing_data, features='text',
                                         method='count')
        # tfidf
        x_testing2 = self.ensemble2.train(self.training_data, self.training_y, self.testing_data, features='text',
                                          method='tfidf')

        """ Test correct data types for countVectorizer"""


        self.assertIsInstance(x_testing, scipy.sparse.csr.csr_matrix)


        """ Test correct data types TfidfVectorizer"""

        self.assertIsInstance(x_testing2, scipy.sparse.csr.csr_matrix)


        """ Test correct behaviour for wrong method"""

        self.assertTrue(
            self.ensemble1.train(self.training_data, self.training_y, self.testing_data, "text", method="ijfsiohf"),1)

    def test_predict(self):
        """ Test Case for Feature Defintion using CountVectorizer"""

        # count
        x_testing = self.ensemble1.train(self.training_data, self.training_y, self.testing_data, features='text',
                                         method='count')
        # tfidf
        x_testing2 = self.ensemble2.train(self.training_data, self.training_y, self.testing_data, features='text',
                                          method='tfidf')

        """ Test correct data types and corrrect range of predicted values (1,0) for predict with countVectorizer"""

        self.assertIsInstance(self.ensemble1.predict(x_testing),
                              np.ndarray)

        self.assertTrue(([0, 1] == np.unique(self.ensemble1.predict(x_testing))).all())

        """ Test correct data types and corrrect range of predicted values (1,0) for predict with countVectorizer"""

        self.assertIsInstance(self.ensemble2.predict(x_testing2),
                              np.ndarray)

        self.assertTrue(([0, 1] == np.unique(self.ensemble2.predict(x_testing2))).all())


if __name__ == "__main__":
    unittest.main()
