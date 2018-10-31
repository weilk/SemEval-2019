from classes import feature_extraction

class number_of_words(feature_extraction):
    
    def run(self,D):
        return {"name":self._name,"features":[len(item.split()) for item in D]}