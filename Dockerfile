FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt update
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN pip3 install tensorflow
RUN pip3 install tensorflow-addons
RUN pip3 install tqdm

WORKDIR /usr/app/src

COPY TripletLoss.py ./
COPY reconFacial.py ./

CMD [ "python3" , "./reconFacial.py" ]