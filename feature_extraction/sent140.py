from classes import feature_extraction
import requests 
import json
import time 

sent_encode = {0:"neg",2:"neutral",4:"pos"}

class sent140(feature_extraction):

    def run(self,D,columns,changes):
    
        if not changes and self.load() == True:
            print("Loaded " + self._name + " from disk")
            return self.saved_data

        for column in columns:
            pos = []
            neg = []
            neutral = []
            l_texts = []
            for line in D[column]:
                line_obj = {}
                line_obj["text"] = str(line)
                l_texts.append(line_obj)
            data = {}
            data["data"] = l_texts  
            print(data)  
            jdata = json.dumps(data)
            try:
                r = requests.post('http://www.sentiment140.com/api/bulkClassifyJson?appid=ingridstoleru@gmail.com', json=jdata)
                print(r)
                j = json.loads(r.text)
                for el in j["data"]:   
                    if el["polarity"] == 0:
                        neg.append(1)
                        pos.append(0)
                        neutral.append(0)
                    if el["polarity"] == 2:
                        neg.append(0)
                        pos.append(0)
                        neutral.append(1)
                    if el["polarity"] == 4:
                        neg.append(0)
                        pos.append(1)
                        neutral.append(0)
  
            except Exception as e:
                print("Error for sent140: {0}".format(e))
          
            self.saved_data['sent140_pos_{}'.format(column)] = pos
            self.saved_data['sent140_neg_{}'.format(column)] = neg
            self.saved_data['sent140_neutral_{}'.format(column)] = neutral        
            
        self.save()
        
        return self.saved_data     
        

