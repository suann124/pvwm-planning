FROM nvidia/cuda:12.2.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install all apt dependencies in one layer (update + install must be combined to avoid stale cache)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglfw3-dev \
    libssl-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    python3-dev \
    python3-pip \
    wget \
    tmux \
    gcc \
    g++ \
    gfortran \
    liblapack-dev \
    pkg-config \
    ipython3 \
    swig \
    libblas-dev \
    libmetis-dev \
    software-properties-common \
    python-is-python3 \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install numpy first with a pinned version compatible with open3d, pandas, and scipy
RUN pip3 install "numpy>=1.23,<1.27" scipy matplotlib open3d


# Install HSL
WORKDIR /
RUN git clone https://github.com/coin-or-tools/ThirdParty-HSL.git
COPY coinhsl /ThirdParty-HSL/coinhsl
WORKDIR /ThirdParty-HSL
RUN ./configure
RUN make
RUN make install

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# Rename libhsl -> libcoinhsl using shell (rename package not available in minimal Ubuntu)
RUN find /usr/local -name "libhsl*" | while read f; do mv "$f" "$(echo "$f" | sed 's/libhsl/libcoinhsl/')"; done


# Install IPOPT
WORKDIR /
RUN git clone https://github.com/coin-or/Ipopt.git
WORKDIR /Ipopt
RUN ./configure
RUN make -j8
RUN make test
RUN make install


WORKDIR /
RUN git clone https://github.com/casadi/casadi.git

WORKDIR /casadi
RUN cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON -DWITH_OPENMP=ON -DWITH_IPOPT=ON -DWITH_HSL=ON .
RUN make
RUN make install


WORKDIR /
RUN pip install warp-lang
RUN wget https://github.com/NVIDIA/warp/releases/download/v1.0.2/warp_lang-1.0.2-py3-none-manylinux2014_x86_64.whl
RUN pip install warp_lang-1.0.2-py3-none-manylinux2014_x86_64.whl

RUN pip install usd-core

COPY wheels /wheels
RUN pip3 install --no-deps /wheels/viser*.whl
RUN pip3 install /wheels/*.whl --no-deps

# expose port for viser/jupyter
EXPOSE 8080

WORKDIR /workspace
COPY . /workspace
RUN pip3 install -e . --no-deps
