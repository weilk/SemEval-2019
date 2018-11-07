import unittest
import pandas
import sys
sys.path.append("..")
import feature_extraction

class TestNumberOfCapitalizedWordsMethods(unittest.TestCase):

    test_dict = {
        "column1": [""],
        "column2": ["This has no long words."],
        "column3": ["This has oooone long word"],
        "column4": ["word word soooo word tooo thaaatttt word"],
        "column5": ["Sssshoo"],
        "column6": ["8888"],
    }

    expected_dict = {
        "column1": [""],
        "column2": ["This has no long words."],
        "column3": ["This has oooone long word"],
        "column4": ["word word soooo word tooo thaaatttt word"],
        "column5": ["Sssshoo"],
        "column6": ["8888"],
        "number_of_elongated_words_column1": [0],
        "number_of_elongated_words_column2": [0],
        "number_of_elongated_words_column3": [1],
        "number_of_elongated_words_column4": [3],
        "number_of_elongated_words_column5": [1],
        "number_of_elongated_words_column6": [0],
    }
    
    test_columns = ["column1","column2","column3","column4","column5","column6"]    
    
    def test_run(self):
        test_dataFrame = pandas.DataFrame(self.test_dict)
        expected_dataFrame = pandas.DataFrame(self.expected_dict)
        feature_extraction.number_of_elongated_words.run(test_dataFrame, self.test_columns)
        pandas.testing.assert_frame_equal(test_dataFrame, expected_dataFrame)

if __name__ == '__main__':
    unittest.main()