from scilus/scilus-freesurfer:2.0.2

WORKDIR /

RUN apt-get -y update --fix-missing 
RUN apt-get install -y git dcm2niix unzip python3-blinker

# Add FSL data
RUN wget https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/archive/2103.0/data_atlases-2103.0.tar.gz
RUN wget https://git.fmrib.ox.ac.uk/fsl/data_standard/-/archive/2208.0/data_standard-2208.0.tar.gz
RUN tar -xzvf data_atlases-2103.0.tar.gz
RUN tar -xzvf data_standard-2208.0.tar.gz
RUN mkdir -p /fsl/data/standard/
RUN mkdir -p /fsl/data/atlases/
RUN mv data_standard-2208.0/* /fsl/data/standard/
RUN mv data_atlases-2103.0/* /fsl/data/atlases/

# Add mirtk
ADD bin/mirtk /usr/local/bin/



# Install nlsam
WORKDIR /
RUN wget https://github.com/samuelstjean/nlsam/releases/download/v0.7/nlsam_v0.7_Linux.zip
RUN unzip nlsam_v0.7_Linux.zip
RUN chmod +x /nlsam_denoising 
RUN mv /nlsam_denoising /usr/local/bin/

# Install Workbench connectom
RUN wget https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip
RUN unzip workbench-linux64-v1.5.0.zip
RUN echo 'export PATH=$PATH:/workbench/bin_linux64' >> ~/.bashrc
ENV PATH /workbench/bin_linux64:$PATH

# Install Unring
RUN git clone https://bitbucket.org/reisert/unring.git
ENV PATH /unring/fsl:$PATH

# Install microbrain
RUN git clone https://github.com/grahamlittlephd/microbrain.git
WORKDIR /microbrain

ADD requirements.txt /microbrain/

ENV PATH /microbrain/bin:$PATH

RUN pip${PYTHON_MOD} install --upgrade pip && \
    pip${PYTHON_MOD} install -U setuptools && \
    pip${PYTHON_MOD} install --ignore-installed -r requirements.txt && \
    pip${PYTHON_MOD} install -e .

# Install
WORKDIR /
RUN git clone https://github.com/LittleBrainLab/meshtrack.git
WORKDIR /meshtrack
RUN pip${PYTHON_MOD} install --ignore-installed -r requirements.txt && \
    pip${PYTHON_MOD} install -e . --use-pep517