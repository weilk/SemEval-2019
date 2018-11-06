import pickle

class data(object):

    def __init__(self,raw,pp,fe,filename=""):
        self._raw = raw
        self._pp = pp 
        self._fe = fe 
        self.D = raw
        self._filename = filename

        self._pp.sort(key=lambda x: x[0]._importance,reverse=True)
        self._fe.sort(key=lambda x: x[0]._importance,reverse=True) 

        if filename == "": 
            for p,c in self._pp:
                self._filename+=p._name
            for f,c in self._fe:
                self._filename+=f._name

        if self.load() == True:
            return

        for p,c in self._pp:
            p.run(self.D["x"],c)
        
        aux = []
        for f,c in self._fe:
            aux.append(f.run(self.D["x"],c))
        self.D["x"] = aux         

        self.save()


    def save(self):
        fd = open("processed_data/" + self._filename + ".data", 'wb')
        pickle.dump(self.__dict__, fd)
        fd.close() 
    
    
    def load(self):
        try:
            fd = open("processed_data/" + self._filename + ".data", 'rb')
            tmp_dict = pickle.load(fd)
            fd.close()
        except IOError:
            return False

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)
        
        return True
