import unittest
import pandas as pd
import sys
sys.path.append("..")
import feature_extraction

class TestNumberExclamationMarks(unittest.TestCase):

    test_dict = {
        "column1": [""],
        "column2": ["!!! day was fully and indeed great!..."],
        "column3": ["!I am hardly pretty purely! not here."],
    }

    expected_dict = {
        "column1": [""],
        "column2": ["!!! day was fully and indeed great!..."],
        "column3": ["!I am hardly pretty purely! not here."],
        "number_of_question_marks_column1": [0],
        "number_of_question_marks_column2": [4],
        "number_of_question_marks_column3": [2],
    }
    
    test_columns = ["column1","column2","column3"]        	

    def test_run(self):
        test_dataFrame = pd.DataFrame(self.test_dict)
        expected_dataFrame = pd.DataFrame(self.expected_dict)
        feature_extraction.number_exclamation_marks.run(test_dataFrame, self.test_columns)
        pd.testing.assert_frame_equal(test_dataFrame, expected_dataFrame)


if __name__ == '__main__':
    unittest.main()