from classes import preprocess

class make_lower_case(preprocess):

    def run(self,D,columns):
        for column in columns:
            D[column] = D[column].apply(str.lower)