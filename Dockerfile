from scilus/scilus-freesurfer:2.0.2

WORKDIR /

# Ubuntu stuff
RUN apt-get -y update --fix-missing 
RUN apt-get install -y git dcm2niix unzip python3-blinker cmake

# ANTS
ARG ANTS_BUILD_NTHREADS
ARG ANTS_INSTALL_PATH
ARG ANTS_VERSION

ENV ANTS_BUILD_NTHREADS=4
ENV ANTS_INSTALL_PATH=/ants
ENV ANTS_VERSION=2.5.1

WORKDIR /
RUN mkdir ants_build && \
    git clone https://github.com/ANTsX/ANTs.git

WORKDIR /ANTs
RUN git fetch --tags && \
    git checkout tags/v${ANTS_VERSION} -b v${ANTS_VERSION}

WORKDIR /ants_build
RUN cmake -DBUILD_SHARED_LIBS=OFF \
          -DUSE_VTK=OFF \
          -DSuperBuild_ANTS_USE_GIT_PROTOCOL=OFF \
          -DBUILD_TESTING=OFF \
          -DRUN_LONG_TESTS=OFF \
          -DRUN_SHORT_TESTS=OFF \
          -DSuperBuild_ANTS_C_OPTIMIZATION_FLAGS="-mtune=native -march=x86-64" \
          -DSuperBuild_ANTS_CXX_OPTIMIZATION_FLAGS="-mtune=native -march=x86-64" \
          -DCMAKE_INSTALL_PREFIX=${ANTS_INSTALL_PATH} \
          ../ANTs && \
    [ -z "$ANTS_BUILD_NTHREADS" ] && \
        { make -j $(nproc --all); } || \
        { make -j ${ANTS_BUILD_NTHREADS}; }

WORKDIR /ants_build/ANTS-build
RUN make install

WORKDIR /
RUN rm -rf /ants_build

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
ADD bin/mirtk /
RUN ./mirtk --appimage-extract
ENV PATH /squashfs-root/usr/bin:$PATH
RUN rm -f /mirtk

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
#RUN git clone https://github.com/LittleBrainLab/microbrain.git
#WORKDIR /microbrain
#RUN git fetch origin pull/11/head && git checkout FETCH_HEAD
WORKDIR /
ADD . /microbrain
RUN rm /microbrain/bin/mirtk

WORKDIR /microbrain
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

#ENV APPIMAGE_EXTRACT_AND_RUN=1