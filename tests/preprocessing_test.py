from src.utils import preprocessing

import unittest


class TestPreprocessing(unittest.TestCase):
    def test_preprocessing(self):
        test_cases = ['This sentence should be converted to lowercase.',
                      'Twitter url should be removed! http://t.co/ba784',
                      'Stopwords like: me myself and i should be removed! :)',
                      '#Hashtag mentions will be deleted by preprocessing',
                      'Special characters like: $[ ] ( }{-.- :( >_< +.+ should be removed',
                      '@mentions to other users should be removed',
                      'Year numbers like 2014 will be remain in text']

        target_results = ['sentenc convert lowercas',
                          'twitter url remov',
                          'stopword like remov',
                          'mention delet preprocess',
                          'special charact like remov',
                          'user remov',
                          'year number like 2014 remain text']

        processed_results = []

        for sentence in test_cases:
            processed_results.append(preprocessing.preprocessing(sentence))

        message = 'Sentences of test cases are processed otherwise than desired'
        self.assertEqual(target_results, processed_results, message)

    def test_preprocessing_restricted(self):
        test_cases = ['This sentence should be converted to lowercase.',
                      'Twitter url should be removed! http://t.co/ba784',
                      'Stopwords like: me myself and i should be removed! :)',
                      '#Hashtag mentions will not deleted by preprocessing',
                      'Special characters like: $[ ] ( }{-.- :( >_< +.+ should be removed',
                      '@mentions to other users should be remain',
                      'year number like 2014 remain text']

        target_results = ['sentenc convert lowercas',
                          'twitter url remov',
                          'stopword like remov',
                          'hashtag mention delet preprocess',
                          'special charact like remov',
                          'mention user remain',
                          'year number like 2014 remain text']

        processed_results = []

        for sentence in test_cases:
            processed_results.append(preprocessing.preprocessing_restricted(sentence))

        message = 'Sentences of test cases are processed otherwise than desired'
        self.assertEqual(target_results, processed_results, message)


if __name__ == "__main__":
    unittest.main()
