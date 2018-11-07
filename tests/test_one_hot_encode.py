import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from preprocess import one_hot_encode

class TestOneHotEncode(unittest.TestCase):

    def test_run(self):
        D = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
        expected = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3],"A_a":[np.uint8(1),np.uint8(0),np.uint8(1)],"A_b":[np.uint8(0),np.uint8(1),np.uint8(0)]})
        expected[["A_a","A_b"]] = expected[["A_a","A_b"]].astype(np.uint8)   
        result = one_hot_encode.run(D,["A"])
        if result is not None:
            D = result
        print(D)
        print(expected)
        pd.testing.assert_frame_equal(D, expected)

if __name__ == '__main__':
    unittest.main()