from classes import feature_extraction
import requests 
import json
import time 

class nltk_sentiment(feature_extraction):

    def run(self,D,columns,changes):
    
        if not changes and self.load() == True:
            print("Loaded " + self._name + " from disk")
            return self.saved_data

        for column in columns:
            pos = []
            neg = []
            neutral = []
            for line in D[column]:
                print(line.encode('UTF-8'))
                j = {}
                while 'probability' not in j.keys():
                    time.sleep(0.1)                 
                    try:                     
                        r = requests.post('http://text-processing.com/api/sentiment/', data = {'text':line.encode('UTF-8')})
                        j = json.loads(r.text)
                        assert 'probability' in j.keys(), 'Missing prob'
                        pos.append(j['probability']['pos'])
                        neg.append(j['probability']['neg'])
                        neutral.append(j['probability']['neutral'])
                    except Exception as e:
                        print("Error occured while requests to nltk sentiment: {0}!\n".format(str(e)))    
                        exit()     
            
            self.saved_data['nltk_sent_pos_{}'.format(column)] = pos
            self.saved_data['nltk_sent_neg_{}'.format(column)] = neg
            self.saved_data['nltk_sent_neutral_{}'.format(column)] = neutral        

        self.save()
        
        return self.saved_data     
        

