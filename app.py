from preprocess import *
from feature_extraction import *
from classes import data
from feature_selection import *
from model import *


raw = {"x":[{"feature1":'''Lorem ipsum dolor sit amet, aliquando incorrupte sed in, prompta repudiandae an eam. Eam labitur denique contentiones ut, cum an virtute scribentur, in velit summo vidisse mel. Scripta partiendo an duo. Has ea dolor nemore graecis, te primis vidisse abhorreant vim. An eam eripuit vituperata omittantur, sed noster blandit cu.

Nullam possit tritani duo no, ut placerat quaerendum sadipscing vis, eos ne virtute ocurreret constituam. No tale fabulas quo. Ea scripta noluisse imperdiet vis, postea euismod impedit an ius, viderer persius at duo. Sea ut novum ceteros, te pertinax dissentiunt pro. Fuisset noluisse partiendo qui at, usu legere hendrerit assueverit te.

Te erat causae eripuit quo, meis tation eos id, eros liber vel te. Te euripidis voluptaria contentiones per, usu illum voluptatum ut, augue nostrum an sit. Mazim disputando ne qui, te per labitur insolens voluptatibus. Mea no tota munere incorrupte. Mollis adolescens definitiones ut duo, cum te dico nibh atqui. Adolescens eloquentiam at per.

Est populo regione definiebas ei, ex partem incorrupte elaboraret usu, his mundi nihil sapientem ad. Everti sanctus in mel, lorem perpetua electram sea at. Ei autem facete sit. Vel id maiorum voluptua forensibus. Persecuti posidonium eu eum, per ex brute molestie, purto nonumy disputando quo et.

Duis posse mediocritatem vis id, cu vis brute mucius aperiri. Munere molestie atomorum per no. Mel te voluptua consectetuer. Libris postea ea est, nam ea harum ornatus atomorum.'''
,"kek":"asdsad"}],"y":[{"target1":"happy", "target2":"sad"}]}
features = []
pp=[(make_lower_case,["feature1"])]
fe=[(number_of_words,["feature1"])]

data_object = data(raw=raw,pp=pp,fe=fe)
print(best_fit.run(data_object.D))
simple_MLP.train(data_object.D)
simple_MLP.test(data_object.D)
print(simple_MLP.forward_pass(data_object.D))



# docker build -t simi2525/ml-env:cpu -f Dockerfile.cpu .
# docker run -it -p 8888:8888 -p 6006:6006  -v ${PWD}/jupyter_notebook_config.py:/root/.jupyter/jupyter_notebook_config.py -v ${PWD}:"/root/SemEval-2019" simi2525/ml-env:cpu
# cd SemEval-2019
# jupyter notebook --allow-root