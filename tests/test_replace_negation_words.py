import unittest
import pandas as pd
import sys
sys.path.append("..")
from preprocess import replace_negation_words

negations = ['no', 'none', 'no one', 'nobody', 'nothing', 'neither', 'nowhere', 'never']

class TestReplaceNegationWords(unittest.TestCase):

    def test_run(self):
        D = pd.DataFrame({'x':["","their result was nothing and nobody liked it. Absolutely never"]})
        expected = pd.DataFrame({"x":["","their result was not and not liked it. Absolutely not"]})

        replace_negation_words.run(D,["x"])
        pd.testing.assert_frame_equal(D, expected)

if __name__ == '__main__':
    unittest.main()