import pickle
import os
import time
import utils
from copy import deepcopy

class data(object):

    def __init__(self,raw,pp,fe,postp,filename=""):
        self._raw = deepcopy(raw)
        self._pp = pp 
        self._fe = fe 
        self._postp = postp
        self.D = raw
        self._filename = filename

        self._pp.sort(key=lambda x: x[0]._importance,reverse=True)
        self._fe.sort(key=lambda x: x[0]._importance,reverse=True) 

        if filename == "": 
            for idx,(p,c) in enumerate(self._pp):
                self._filename+="_p"+str(p._id)
            for idx,(f,c) in enumerate(self._fe):
                self._filename+="_f"+str(f._id)
            for idx,(post,c) in enumerate(self._postp):
                self._filename+="_post"+str(post._id)

        if self.load() == True:
            utils.output_emocontext.extend(["label_happy","label_angry","label_sad","label_others"])
            print("Loaded from disk")
            return
        
        self._filename = ""
        big_start = time.time()
        for idx,(p,c) in enumerate(self._pp):
            self._filename+="_p"+str(p._id)
            if self.load() == True:
                print("Loaded "+p._name+" from disk")
                continue
            print("PP: {}, {}".format(p,c))
            start = time.time()
            result = p.run(self.D,c)
            if result is not None:
                self.D = result
            print("Taken: {}".format(time.time() - start))
            self.save()

        for idx,(f,c) in enumerate(self._fe):
            self._filename+="_f"+str(f._id)
            if self.load() == True:
                print("Loaded "+f._name+" from disk")
                continue
            print("FE: {}, {}".format(f,c))
            start = time.time()
            result = f.run(self.D,c)
            if result is not None:
                self.D = result
            print("Taken: {}".format(time.time() - start))
            self.save()

        for idx,(post,c) in enumerate(self._postp):
            self._filename+="_post"+str(post._id)
            if self.load() == True:
                print("Loaded "+post._name+" from disk")
                continue
            print("POSTP: {}, {}".format(post,c))
            start = time.time()
            result = post.run(self.D,c)
            if result is not None:
                self.D = result
            print("Taken: {}".format(time.time() - start))
            self.save()
        
        print("Total time for building the database: {}".format(time.time() - big_start))
        
        self.save()


    def save(self):
        try:
            if not os.path.exists("processed_data"):
                os.makedirs("processed_data")
            fd = open("processed_data/" + self._filename + ".data", 'wb')
            pickle.dump(self.__dict__, fd)
            fd.close() 
        except IOError:
            return False
        
    
    def load(self):
        try:
            fd = open("processed_data/" + self._filename + ".data", 'rb')
            tmp_dict = pickle.load(fd)
            fd.close()
        except IOError:
            return False

        filename = self._filename
        raw = deepcopy(self._raw)
        pp = self._pp
        fe = self._fe
        postp = self._postp

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)

        self._raw=raw
        self._pp=pp
        self._fe=fe
        self._postp = postp
        self._filename=filename
        
        return True
