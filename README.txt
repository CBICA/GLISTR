
  Section of Biomedical Image Analysis
  Center for Biomedical Image Computing and Analytics
  Department of Radiology
  University of Pennsylvania
  3700 Hamilton Walk, Floor 7
  Philadelphia, PA 19104

  Web:   https://www.cbica.upenn.edu/sbia/
  Email: software at cbica.upenn.edu

  Copyright (c) 2016 University of Pennsylvania. All rights reserved.
  See https://www.cbica.upenn.edu/sbia/software/license.html or COPYING file.

 

INTRODUCTION
============

  GLioma Image SegmenTation and Registration (GLISTR) [1,2,3] is a software package 
  designed for simultaneously segmenting brain scans of glioma patients and 
  registering these scans to a normal atlas.

  Some typical applications of GLISTR include,

    - Labeling entire brain regions of glioma patients;
    - Mapping gliomas into the healthy atlas space;
    - Estimating parameters of the tumor growth model.

  GLISTR is implemented as a command-line tool. It is semi-automatic and requires 
  minimal user initializations. Users could use the visual interface called 
  BrainTumorViewer to easily make initializations and a script for the execution. 
  As a results, GLISTR will output a label map, a mapping between atlas and input, 
  tumor parameters, etc.

  

PACKAGE OVERVIEW
================

  - config/             Package configuration files.
  - data/               The atlas used by GLISTR.
  - doc/                Software documentation such as the software manual.
  - example/            Example data files.
  - lib/                Library source code files.
  - src/                Main source code files.
  - AUTHORS.txt         A list of the people who contributed to this software.
  - ChangeLog.txt       A log of changes between versions.
  - CMakeLists.txt      Root CMake configuration file.
  - COPYING.txt         The copyright and license notices.
  - INSTALL.txt         Build and installation instructions.
  - README.txt          This readme file.



DOCUMENTATION
=============

  See the software manual for details on the software including a demonstration
  of how to apply the software tools provided by this package.



INSTALLATION
============

  See http://www.cbica.upenn.edu/sbia/software/glistr/installation.html or
  the installation section of the software manual.

  
  
LICENSING
=========

  See http://www.cbica.upenn.edu/sbia/software/license.html or COPYING file.



REFERENCES
==========

  [1] A. Gooya, K.M. Pohl, M. Bilello, L. Cirillo, G. Biros, E.R. Melhem, C. Davatzikos, 
      "GLISTR: Glioma Image Segmentation and Registration", IEEE Trans. Med. Imaging 31(10): 
      1941-1954 (2012) 
      http://dx.doi.org/10.1109/TMI.2012.2210558

  [2] D. Kwon, R.T. Shinohara, H. Akbari, C. Davatzikos, "Combining Generative Models for 
      Multifocal Glioma Segmentation and Registration", In: Proc. MICCAI (1): 763-770 (2014) 
      http://dx.doi.org/10.1007/978-3-319-10404-1_95
  
  [3] http://www.cbica.upenn.edu/sbia/software/glistr/
