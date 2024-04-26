from scilus/scilus-freesurfer:1.6.0

WORKDIR /

RUN apt-get -y update
RUN apt-get install -y git dcm2niix

# Install Unring
RUN git clone https://bitbucket.org/reisert/unring.git
ENV PATH=/unring/fsl:$PATH

# Install microbrain
RUN git clone https://github.com/grahamlittlephd/microbrain.git
WORKDIR /microbrain

ADD requirements.txt /microbrain/

ENV PATH=/microbrain/bin:$PATH

RUN pip${PYTHON_MOD} install --upgrade pip && \
    pip${PYTHON_MOD} install -U setuptools && \
    pip${PYTHON_MOD} install -r requirements.txt --ignore-installed blinker && \
    pip${PYTHON_MOD} install -e .