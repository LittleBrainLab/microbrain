from scilus/scilus-freesurfer:1.6.0

WORKDIR /

RUN apt-get -y update --fix-missing 
RUN apt-get install -y git dcm2niix unzip python3-blinker

# Install Workbench connectom
RUN wget https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip
RUN unzip workbench-linux64-v1.5.0.zip
RUN echo 'export PATH=$PATH:/workbench/bin_linux64' >> ~/.bashrc

# Install Unring
RUN git clone https://bitbucket.org/reisert/unring.git
ENV PATH=/unring/fsl:$PATH

# Install microbrain
RUN git clone https://github.com/grahamlittlephd/microbrain.git
WORKDIR /microbrain

# ADD requirements.txt /microbrain/

ENV PATH=/microbrain/bin:$PATH

RUN pip${PYTHON_MOD} install --upgrade pip && \
    pip${PYTHON_MOD} install -U setuptools && \
    pip${PYTHON_MOD} install --ignore-installed -r requirements.txt && \
    pip${PYTHON_MOD} install -e .