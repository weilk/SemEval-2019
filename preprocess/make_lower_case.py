from classes import preprocess

class make_lower_case(preprocess):

    def run(self,D,columns):
        for row in D:
            for feature in columns:
                if feature in columns:
                    row[feature] = row[feature].lower()
        # return D
        # return [{feature:row[feature].lower() for feature in row if feature in columns}for row in D] 