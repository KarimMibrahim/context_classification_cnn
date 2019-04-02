#FROM debian:stretch
FROM tensorflow/tensorflow:1.12.0-rc2-py3

#FROM nvidia/cuda:9.1-runtime-ubuntu16.04

# install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git


# clone inside image github repository
RUN git clone https://github.com/KarimMibrahim/context_classification_cnn /src_code/repo
ENV PYTHONPATH=$PYTHONPATH:/src_code/repo


# Downgrade to cudatoolkit 9.0 for compatibility reasons
#RUN conda install -y -c conda-forge cudatoolkit=9.0

# install requirements
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN rm requirements.txt

ENV CUDA_VISIBLE_DEVICES=3

#ENTRYPOINT ["bash" ,"-c","source $DATARD_PATH/research_setenv.sh.dev;$SHELL"]