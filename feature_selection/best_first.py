from classes import feature_selection
import random

class best_fit(feature_selection):
    
    def run(self,D):
        return random.choice(D["x"])