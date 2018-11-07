import unittest
import pandas as pd
import sys
sys.path.append("..")
import feature_extraction

class TestNumberNegationWords(unittest.TestCase):

    test_dict = {
        "column1": [""],
        "column2": ["THE day was fully and indeed great..."],
        "column3": ["I am hardly pretty purely not here."],
        "column4": ["Here it goes, nice one."],
    }

    expected_dict = {
        "column1": [""],
        "column2": ["THE day was fully and indeed great..."],
        "column3": ["I am hardly pretty purely not here."],
        "column4": ["Here it goes, nice one."],
        "number_of_boosting_words_column1": [0],
        "number_of_boosting_words_column2": [2],
        "number_of_boosting_words_column3": [3],
        "number_of_boosting_words_column4": [0],
    }
    
    test_columns = ["column1","column2","column3","column4"]        	

    def test_run(self):
        test_dataFrame = pd.DataFrame(self.test_dict)
        expected_dataFrame = pd.DataFrame(self.expected_dict)
        feature_extraction.number_boosting_words.run(test_dataFrame, self.test_columns)
        pd.testing.assert_frame_equal(test_dataFrame, expected_dataFrame)


if __name__ == '__main__':
    unittest.main()