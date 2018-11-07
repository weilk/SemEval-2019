import unittest
import pandas as pd
import sys
sys.path.append("..")
from preprocess import tokenization

class TestTokenization(unittest.TestCase):

    def test_run(self):
        D = pd.DataFrame({'x':["", "their result was nothing and nobody liked it. Absolutely never"]})
        expected = pd.DataFrame({'x':["", "their result was nothing and nobody liked it. Absolutely never"], "tokenize_x":[[], ["their", "result", "was", "nothing", "and", "nobody", "liked", "it", ".", "Absolutely", "never"]]})

        tokenization.run(D,["x"])
        pd.testing.assert_frame_equal(D, expected)

if __name__ == '__main__':
    unittest.main()