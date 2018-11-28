import pickle
import os
import time
import utils
from copy import deepcopy
import pandas as pd

class data(object):

    def __init__(self,raw,pp,fe,postp,fs,filename=""):
        self._raw = deepcopy(raw)
        self._pp = pp 
        self._fe = fe 
        self._postp = postp
        self._fs = fs
        self.D = raw
        self._filename = filename

        self._pp.sort(key=lambda x: x[0]._importance,reverse=True)
        self._fe.sort(key=lambda x: x[0]._importance,reverse=True) 
        self._postp.sort(key=lambda x: x[0]._importance,reverse=True)
        self._fs.sort(key=lambda x: x[0]._importance,reverse=True) 

        self._filename = ""
        changes = False
        big_start = time.time()
        for idx,(p,c) in enumerate(self._pp):
            self._filename+="_p"+str(p._id)
            if self.load() == True:
                if p._name == "one_hot_encode":
                    utils.output_emocontext.extend(["label_happy","label_angry","label_sad","label_others"])
                print("Loaded "+p._name+" from disk")
                continue
                
            changes = True           
            print("PP: {}, {}".format(p,c))
            start = time.time()
            result = p.run(self.D,c)
            if result is not None:
                self.D = result
            print("Taken: {}".format(time.time() - start))
            self.save()

        for idx,(f,c) in enumerate(self._fe):
            self._filename+="_f"+str(f._id)
            
            print("FE: {}, {}".format(f,c))
            start = time.time()
            result = f.run(self.D,c,changes)
            if result is not None:
                self.D = pd.concat([self.D, result], axis=1, sort=False)
            print("Taken: {}".format(time.time() - start))

        for idx,(post,c) in enumerate(self._postp):
            self._filename+="_post"+str(post._id)
            if not changes and self.load() == True:
                print("Loaded "+post._name+" from disk")
                continue
            print("POSTP: {}, {}".format(post,c))
            start = time.time()
            result = post.run(self.D,c)
            if result is not None:
                self.D = result
            print("Taken: {}".format(time.time() - start))
            self.save()

        for idx,(fs,) in enumerate(self._fs):
            self._filename+="_fs"+str(fs._id)
            if not changes and self.load() == True:
                print("Loaded "+fs._name+" from disk")
                continue
            print("FS: {}".format(fs))
            start = time.time()
            result = fs.run(self.D)
            if result is not None:
                self.D = result
            print("Taken: {}".format(time.time() - start))
            self.save()
        
        print("Total time for building the database: {}".format(time.time() - big_start))
        

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
        fs = self._fs

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)

        self._raw=raw
        self._pp=pp
        self._fe=fe
        self._postp = postp
        self._fs=fs
        self._filename=filename
        
        return True
