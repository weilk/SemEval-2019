import unittest
import pandas as pd
import sys
sys.path.append("..")
import feature_extraction

class TestNumberNegationWords(unittest.TestCase):

    test_dict = {
        "column1": [""],
        "column2": ["THE day was MORE than..."],
        "column3": ["I am not here."],
        "column4": ["Here it goes, notredam."],
        "column5": ["There there not."],
    }

    expected_dict = {
        "column1": [""],
        "column2": ["THE day was MORE than..."],
        "column3": ["I am not here."],
        "column4": ["Here it goes, notredam."],
        "column5": ["There there not."], 
        "number_of_negated_words_column1": [0],
        "number_of_negated_words_column2": [0],
        "number_of_negated_words_column3": [1],
        "number_of_negated_words_column4": [0],
        "number_of_negated_words_column5": [1],
    }
    
    test_columns = ["column1","column2","column3","column4","column5"]        	

    def test_run(self):
        test_dataFrame = pd.DataFrame(self.test_dict)
        expected_dataFrame = pd.DataFrame(self.expected_dict)
        feature_extraction.number_negation_words.run(test_dataFrame, self.test_columns)
        pd.testing.assert_frame_equal(test_dataFrame, expected_dataFrame)


if __name__ == '__main__':
    unittest.main()