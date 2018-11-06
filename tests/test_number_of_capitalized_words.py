import unittest
import pandas
import sys
sys.path.append("..")
import feature_extraction

class TestNumberOfCapitalizedWordsMethods(unittest.TestCase):

    test_dict = {
        "column1": [""],
        "column2": ["THE day was MORE than..."],
        "column3": ["THIS"],
        "column4": ["ThhE daY, Wooord"],
        "column5": ["ana are mere"],
        "column6": ["THIs WAs ONE Day"],
        "column7": ["THIs WAs ONE DAAY,,,,,!!!!"],
        "column8": ["THIs was,,,,,,,,!!!!"],
    }

    expected_dict = {
        "column1": [""],
        "column2": ["THE day was MORE than..."],
        "column3": ["THIS"],
        "column4": ["ThhE daY, Wooord"],
        "column5": ["ana are mere"],
        "column6": ["THIs WAs ONE Day"],
        "column7": ["THIs WAs ONE DAAY,,,,,!!!!"],
        "column8": ["THIs was,,,,,,,,!!!!"],
        "number_of_capitalized_words_column1": [0],
        "number_of_capitalized_words_column2": [2],
        "number_of_capitalized_words_column3": [1],
        "number_of_capitalized_words_column4": [0],
        "number_of_capitalized_words_column5": [0],
        "number_of_capitalized_words_column6": [1],
        "number_of_capitalized_words_column7": [2],
        "number_of_capitalized_words_column8": [0],
    }
    
    test_columns = ["column1","column2","column3","column4","column5","column6","column7","column8"]    
    
    def test_run(self):
        test_dataFrame = pandas.DataFrame(self.test_dict)
        expected_dataFrame = pandas.DataFrame(self.expected_dict)
        feature_extraction.number_of_capitalized_words.run(test_dataFrame, self.test_columns)
        pandas.testing.assert_frame_equal(test_dataFrame, expected_dataFrame)

if __name__ == '__main__':
    unittest.main()