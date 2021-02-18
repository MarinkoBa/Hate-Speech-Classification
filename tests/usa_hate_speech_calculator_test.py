from src.utils.usa_hate_speech_calculator import calculate_hate_speech_ratio

import pandas as pd
import numpy as np
import unittest


class TestDataCollection(unittest.TestCase):
    def setUp(self):
        self.d = {'city_name': ["ab","ab","ab","ab", "c","c","c", "def","def"], 
                  'text': ["bla bla bal", "bla bla bal","bla bla bal","bla bla bal","bla bla bal","bla bla bal","bla bla bal","bla bla bal","bla bla bal",]}
        self.df = pd.DataFrame(data=self.d)
        
        self.n=np.array([1, 0, 1,1,0,0,0,1,0])

    def test_load_data(self):
        a=calculate_hate_speech_ratio(self.df, self.n)
        
        self.assertEqual(a[0],3/4)
        self.assertEqual(a[1],0)
        self.assertEqual(a[2],1/2)
        


if __name__ == "__main__":
    unittest.main()
