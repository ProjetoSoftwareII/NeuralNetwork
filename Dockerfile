FROM ubuntu:latest

RUN apt update
RUN apt install python3 -y
RUN pip3 install tensorflow tensorflow-gpu -y
RUN pip3 install tensorflow-addons
WORKDIR /usr/app/src

COPY reconFacial.py ./

CMD [ "python3" , "./reconFacial.py" ]