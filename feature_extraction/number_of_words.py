from classes import feature_extraction

class number_of_words(feature_extraction):
    
    def run(self,D,columns = []):
        copied_data = list(D)
        new_row = {}
        for row in copied_data:
            for feature in columns:
                if feature in columns:
                    new_row[self._name] = len(row[feature].split())
        return new_row