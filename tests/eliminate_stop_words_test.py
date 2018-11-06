import unittest
import pandas as pd
import sys
sys.path.append("..")
from preprocess import eliminate_stop_words

class TestEliminateStopWords(unittest.TestCase):

    def test_run(self):
        D = pd.DataFrame({'x':["","their result was good and bad"],"y":["","result good bad"]})
        eliminate_stop_words.run(D,["x"])
        pd.testing.assert_series_equal(D["x"],D["y"])

if __name__ == '__main__':
    unittest.main()