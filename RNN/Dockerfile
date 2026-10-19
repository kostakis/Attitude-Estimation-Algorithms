# Minimal Docker image for LSTM training
# Base: Debian bookworm-slim (~74MB, glibc-based for LibTorch compatibility)
FROM debian:bookworm-slim

# Install minimal dependencies: g++, make, wget, unzip, Eigen
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    wget \
    unzip \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install LibTorch (CPU version, C++11 ABI)
WORKDIR /opt
RUN wget -q https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip && \
    unzip -q libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

WORKDIR /tmp
RUN wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.zip && \ 
    unzip -q eigen-3.4.1.zip && \
    cp -r */Eigen /usr/include && \
    rm -rf eigen-3.4.1* eigen-3.4.1.zip

# Set library path for LibTorch shared libraries
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# Create app directory (WORKDIR creates it if it doesn't exist)
WORKDIR /app

# Needed when running the evalution to save the csv file
RUN mkdir /app/Results

COPY Data/ ./Data/
COPY include/ ./include/
COPY RNN/Makefile Makefile

RUN sed -i 's|LIBTORCH_PATH = .*|LIBTORCH_PATH = /opt/libtorch|g' Makefile

# 4. Source files (change frequently) - copied last
COPY RNN/*.cpp RNN/*.h ./

# Compile the LSTM application
WORKDIR /app
RUN make all -j

# Default command: run the trained model
CMD ["./lstm.out", "--epochs", "10"]
