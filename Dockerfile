FROM centos:6.10

MAINTAINER CBICA_UPenn software@cbica.upenn.edu

#update
RUN yum update -y

#general dependencies
RUN yum install -y \
    gcc gcc-c++ \
    wget \
    cmake \
    git-core \
    lapack \
    lapack-devel \
    qt \
    unzip \
    tcl \
    tcl-devel \
    tk \
    tk-devel \
    fftw \
    fftw-devel \
    mpich \
    mpich-devel

#ITK-4.7.0
RUN wget https://github.com/InsightSoftwareConsortium/ITK/archive/v4.7.0.zip; \
    unzip v4.7.0.zip; \
    cd ITK-4.7.0; \
    mkdir bin; \
    cd bin; \
    cmake \
    -DITK_IMAGE_BEHAVES_AS_ORIENTED_IMAGE:BOOL=TRUE \
    -DITK_USE_OPTIMIZED_REGISTRATION_METHODS:BOOL=TRUE \
    -DITK_USE_ORIENTED_IMAGE_DIRECTION_METHODS:BOOL=TRUE \
    -DITK_USE_REGION_VALIDATION_IN_ITERATORS:BOOL=TRUE \
    -DITK_USE_REVIEW:BOOL=TRUE \
    -USE_FFTWF=ON \
    -DITK_USE_PATENTED:BOOL=TRUE \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE:STRING=RELEASE \
    -Wno-dev ..; \
    make; \
    export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:'pwd'; \
    rm -rf /v4.7.0.zip

#PETSc-3.5.2
RUN wget https://www.mcs.anl.gov/petsc/mirror/release-snapshots/petsc-3.5.2.tar.gz; \
    tar xzf petsc-3.5.2.tar.gz; \
    cd petsc-3.5.2; \
    ./configure --with-mpi=0 --download-f2cblaslapack=1 --prefix=./build; \
    make PETSC_DIR=/petsc-3.5.2 PETSC_ARCH=arch-linux2-c-debug all; \
    make PETSC_DIR=/petsc-3.5.2 PETSC_ARCH=arch-linux2-c-debug install; \
    rm -rf /petsc-3.5.2.tar.gz

#BTMCS-1.2.1
RUN wget https://github.com/CBICA/BTMCS/archive/1.2.1.zip; \
    unzip 1.2.1.zip; \
    mkdir btmcs-1.2.1-build; \
    cd btmcs-1.2.1-build; \
    cmake \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/btmcs-1.2.1-build \
    -DITK_DIR=/ITK-4.7.0/bin \
    -DMAKE_EXECUTABLE=/usr/bin/make \
    -DPETSC_ARCH=/arch-linux2-c-debug \
    -DPETSC_CURRENT=ON \
    -DPETSC_DIR=/petsc-3.5.2 \
    -DMPI_COMPILER=/usr/lib64/mpich/bin/mpicxx \
    ../BTMCS-1.2.1; \
    make; \
    make install; \
    rm -rf /1.2.1.zip

#HOPSPACK-2.0.2
RUN wget https://dakota.sandia.gov/sites/default/files/hopspack-2.0.2-src.tar.gz; \
    gzip -d hopspack-2.0.2-src.tar.gz; \
    tar xf hopspack-2.0.2-src.tar; \
    mkdir hopspack-2.0.2-build && cd hopspack-2.0.2-build; \
    mkdir serial mt mpi install; \
    cd serial; \
    cmake ../../hopspack-2.0.2-src \
    -DCMAKE_INSTALL_PREFIX=/hopspack-2.0.2-build/install \
    -DCMAKE_BUILD_TYPE=Release \
    -DLAPACK_LIBS=/usr/lib64/liblapack.so \
    -Ddebug=ON \
    -Dlapack=ON \
    -Dmpi=OFF \
    -Dmt=OFF; \ 
    make; \
    cd ../mt; \
    cmake ../../hopspack-2.0.2-src \
    -DCMAKE_INSTALL_PREFIX=/hopspack-2.0.2-build/install \
    -DCMAKE_BUILD_TYPE=Release \
    -DLAPACK_LIBS=/usr/lib64/liblapack.so \
    -Ddebug=ON \
    -Dlapack=ON \
    -Dmpi=OFF \
    -Dmt=ON; \
    make; \
    cd ../mpi; \
    cmake ../../hopspack-2.0.2-src \
    -DCMAKE_INSTALL_PREFIX=/hopspack-2.0.2-build/install \
    -DCMAKE_BUILD_TYPE=Release \
    -DLAPACK_LIBS=/usr/lib64/liblapack.so \
    -Ddebug=ON \
    -Dlapack=ON \
    -Dmpi=ON \
    -Dmt=OFF \
    -DMPI_COMPILER=/usr/lib64/mpich/bin/mpicxx; \
    make; \
    cd ../install; \
    mkdir bin; \
    cp -r ../serial/{examples,LICENSE,src,test} .; \
    cp -r ../..//hopspack-2.0.2-src/doc/ .; \
    cp ../*/HOPSPACK_main_* bin/; \
    rm -rf hopspack-2.0.2-src.tar

#FSL-4.1.5
RUN wget https://github.com/jminock/fsl-4.1.5/archive/master.zip; \
    unzip master.zip; \
    chmod +x fsl-4.1.5-master/bin/flirt; \
    chmod +x fsl-4.1.5-master/bin/convert_xfm; \
    rm -rf master.zip
    

#GLISTR-3.1.0
RUN wget https://github.com/CBICA/GLISTR/archive/3.1.0.zip; \
    unzip 3.1.0.zip; \
    cd GLISTR-3.1.0; \
    mkdir bin; \
    cd bin; \
    cmake \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_BUILD_TYPE=Release \
    -DITK_DIR=/ITK-4.7.0/bin \
    -DCUDA_SEPARABLE_COMPILATION=OFF \
    -DBUILD_TESTING=OFF \
    ..; \
    make; \
    make install; \
    rm -rf /3.1.0.zip
    
ENV FSLDIR=/fsl-4.1.5-master/
ENV FSLOUTPUTTYPE=NIFTI_PAIR
ENV FSLMULTIFILEQUIT=TRUE
#ENV FLIRT_PATH=/GLISTR-3.1.0/bin/install/bin
#ENV MODULE_PATH=/GLISTR-3.1.0/bin/install/bin
#copy runtime dependencies into GLISTR 
RUN cp hopspack-2.0.2-build/install/bin/HOPSPACK_main_threaded GLISTR-3.1.0/bin/install/bin/; \
    cp fsl-4.1.5-master/bin/flirt GLISTR-3.1.0/bin/install/bin/flirt; \
    cp fsl-4.1.5-master/bin/convert_xfm GLISTR-3.1.0/bin/install/bin/convert_xfm; \
    cp btmcs-1.2.1-build/bin/ForwardSolverDiffusion GLISTR-3.1.0/bin/install/bin/

#Run GLISTR-3.1.0
ENTRYPOINT ["/GLISTR-3.1.0/bin/install/bin/GLISTR"]
