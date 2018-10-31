from classes import preprocess

class make_lower_case(preprocess):

    def run(self,D):
        return [item.lower() for item in D]