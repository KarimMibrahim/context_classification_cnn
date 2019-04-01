FROM nvidia/cuda:9.1-runtime-ubuntu16.04

# env variable for dzr_audio
ENV DATARD_PATH=/mnt/nfs/analysis DZR_MUSIC_DB=/mnt/music/output/mp3_128/
ENV DZR_RESEARCH_CODE=/srv/workspace/research
ENV PYTHONPATH=$PYTHONPATH:$DZR_RESEARCH_CODE

WORKDIR $DZR_RESEARCH_CODE

# install anaconda
ENV PATH /opt/conda/bin:$PATH
COPY install_miniconda.sh .
RUN bash ./install_miniconda.sh && rm install_miniconda.sh

RUN conda install -y python=3.6.6

RUN git clone https://github.deezerdev.com/Research/dzr_utils.git $DZR_RESEARCH_CODE/dzr_utils
RUN cd $DZR_RESEARCH_CODE/dzr_utils && git checkout python3

RUN conda install -c conda-forge ipdb

RUN conda install scikit-learn

# install tensorflow for GPU
RUN conda install -y -c conda-forge tensorflow-gpu==1.12.0

# Downgrade to cudatoolkit 9.0 for compatibility reasons
RUN conda install -y -c conda-forge cudatoolkit=9.0

RUN conda install -y -c anaconda tqdm
RUN conda install -y -c anaconda pandas==0.24.1

ENV PYTHONPATH=$PYTHONPATH:$DZR_RESEARCH_CODE
ENV CUDA_VISIBLE_DEVICES=3

ENTRYPOINT ["bash" ,"-c","source $DATARD_PATH/research_setenv.sh.dev;$SHELL"]

# add deezer user
RUN groupadd -g 100000 users_deezer
RUN useradd -r -m -s /bin/bash -u 100000 deezer
USER deezer:users_deezer