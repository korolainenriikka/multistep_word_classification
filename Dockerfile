FROM continuumio/miniconda3:latest

RUN pip install mlflow==1.17.0 \
    && pip install scikit-learn==0.24.0 \
    && pip install lxml==4.6.3 \
    && pip install click==8.0.1
RUN apt-get update
RUN apt-get -y install apt-transport-https ca-certificates curl gnupg2 software-properties-common
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"

RUN apt-get update
RUN apt-get -y install docker-ce



