FROM ubuntu:latest

RUN apt update
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN pip3 install tensorflow
RUN pip3 install tensorflow-addons

WORKDIR /usr/app/src

COPY reconFacial.py ./

CMD [ "python3" , "./reconFacial.py" ]