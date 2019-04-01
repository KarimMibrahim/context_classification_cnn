FROM debian:stretch

#FROM nvidia/cuda:9.1-runtime-ubuntu16.04

# install anaconda
# clone inside image github repository
RUN git clone https://github.com/KarimMibrahim/context_classification_cnn /src_code/repo
ENV PYTHONPATH=$PYTHONPATH:/src_code/repo

ENV PATH /opt/conda/bin:$PATH
COPY install_miniconda.sh .
RUN bash ./install_miniconda.sh && rm install_miniconda.sh

# install requirements
RUN conda install -y python=3.6.6

RUN conda install -c conda-forge ipdb

RUN conda install scikit-learn

# install tensorflow for GPU
RUN conda install -y -c conda-forge tensorflow-gpu==1.12.0

# Downgrade to cudatoolkit 9.0 for compatibility reasons
RUN conda install -y -c conda-forge cudatoolkit=9.0

RUN conda install -y -c anaconda tqdm
RUN conda install -y -c anaconda pandas==0.24.1

#ENV PYTHONPATH=$PYTHONPATH:$DZR_RESEARCH_CODE
ENV CUDA_VISIBLE_DEVICES=3

#ENTRYPOINT ["bash" ,"-c","source $DATARD_PATH/research_setenv.sh.dev;$SHELL"]