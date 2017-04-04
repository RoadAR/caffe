---
title: "Installation: RHEL / Fedora / CentOS"
---

# RHEL / Fedora / CentOS Installation

**General dependencies**

    sudo yum install protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel

**Remaining dependencies, recent OS**

    sudo yum install gflags-devel lmdb-devel

**CUDA**: Install via the NVIDIA package instead of `yum` to be certain of the library and driver versions.
Install the library and latest driver separately; the driver bundled with the library is usually out-of-date.
    + CentOS/RHEL/Fedora:

**BLAS**: install ATLAS by `sudo yum install atlas-devel` or install OpenBLAS or MKL for better CPU performance. For the Makefile build, uncomment and set `BLAS_LIB` accordingly as ATLAS is usually installed under `/usr/lib[64]/atlas`).

**Python** (optional): if you use the default Python you will need to `sudo yum install` the `python-devel` package to have the Python headers for building the pycaffe wrapper.

Continue with [compilation](installation.html#compilation).
