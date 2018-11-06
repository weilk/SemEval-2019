import unittest
import pandas as pd
import sys
sys.path.append("..")
from preprocess import eliminate_stop_words

class TestEliminateStopWords(unittest.TestCase):

    def test_run(self):
        D = pd.DataFrame({'x':["","their result was good and bad"]})
        expected = pd.DataFrame({"x":["","result good bad"]})

        eliminate_stop_words.run(D,["x"])
        pd.testing.assert_frame_equal(D,expected)

if __name__ == '__main__':
    unittest.main()