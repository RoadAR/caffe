---
title: "Installation: Ubuntu"
---

# Ubuntu Installation

**General dependencies**

    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
    sudo apt-get install --no-install-recommends libboost-all-dev

**CUDA**: Install by `apt-get` or the NVIDIA `.run` package.
The NVIDIA package tends to follow more recent library and driver versions, but the installation is more manual.
If installing from packages, install the library and latest driver separately; the driver bundled with the library is usually out-of-date.
This can be skipped for CPU-only installation.

**BLAS**: install ATLAS by `sudo apt-get install libatlas-base-dev` or install OpenBLAS or MKL for better CPU performance.

**Python** (optional): if you use the default Python you will need to `sudo apt-get install` the `python-dev` package to have the Python headers for building the pycaffe interface.

**Compatibility notes, 16.04**

CUDA 8 is required on Ubuntu 16.04.

**Remaining dependencies, 14.04**

Everything is packaged in 14.04.

    sudo apt-get install libgflags-dev liblmdb-dev

Continue with [compilation](installation.html#compilation).
