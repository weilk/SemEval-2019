FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install wget make build-essential checkinstall -y

#RUN apt-get install build-essential checkinstall -y
#RUN apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev -y
RUN cd /usr/src
RUN wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz
RUN tar xzf Python-3.7.0.tgz

WORKDIR "Python-3.7.0"
RUN ./configure
RUN make install

RUN pip3 install tensorflow
WORKDIR "~/"
CMD /bin/bash
