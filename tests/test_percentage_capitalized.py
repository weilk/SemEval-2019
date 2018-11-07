import unittest
import pandas as pd
import sys
sys.path.append("..")
import feature_extraction
import numpy as np

class TestPercentageCapitalized(unittest.TestCase):

    test_dict = {
        "column1": [""],
        "column2": ["DAY was fully and INDEED great"],
        "column3": ["I am hardly pretty PURELY NOT HERE."],
    }

    expected_dict = {
        "column1": [""],
        "column2": ["DAY was fully and INDEED great"],
        "column3": ["I am hardly pretty PURELY NOT HERE."],
        "percentage_of_capitalized_column1": [np.nan],
        "percentage_of_capitalized_column2": [2/6],
        "percentage_of_capitalized_column3": [4/7],
    }
    
    test_columns = ["column1","column2","column3"]        	

    def test_run(self):
        print(self.test_dict)
        print(self.expected_dict)
        test_dataFrame = pd.DataFrame(self.test_dict)
        expected_dataFrame = pd.DataFrame(self.expected_dict)
        feature_extraction.percentage_capitalized.run(test_dataFrame, self.test_columns)
        pd.testing.assert_frame_equal(test_dataFrame, expected_dataFrame)


if __name__ == '__main__':
    unittest.main()