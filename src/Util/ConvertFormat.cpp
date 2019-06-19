///////////////////////////////////////////////////////////////////////////////////////
// ConvertFormat.cpp
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014 University of Pennsylvania. All rights reserved.
// See http://www.cbica.upenn.edu/sbia/software/license.html or COYPING file.
//
// Contact: SBIA Group <sbia-software at uphs.upenn.edu>
///////////////////////////////////////////////////////////////////////////////////////

#include "../stdafx.h"
#include <stdio.h>
#include <string.h>
#include "../MyUtils.h"
#include "../Volume.h"
//
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"
#include "itkNiftiImageIO.h"


// float: 0, double: 1, byte: 2, short: 3, int: 4
typedef itk::ImageFileReader<itk::Image<float, 3> > fImageReaderType;
typedef itk::ImageFileWriter<itk::Image<float, 3> > fImageWriterType;
typedef itk::ImageFileReader<itk::Image<double, 3> > dImageReaderType;
typedef itk::ImageFileWriter<itk::Image<double, 3> > dImageWriterType;
typedef itk::ImageFileReader<itk::Image<unsigned char, 3> > bImageReaderType;
typedef itk::ImageFileWriter<itk::Image<unsigned char, 3> > bImageWriterType;
typedef itk::ImageFileReader<itk::Image<short, 3> > sImageReaderType;
typedef itk::ImageFileWriter<itk::Image<short, 3> > sImageWriterType;
typedef itk::ImageFileReader<itk::Image<int, 3> > iImageReaderType;
typedef itk::ImageFileWriter<itk::Image<int, 3> > iImageWriterType;


void version()
{
	printf("==========================================================================\n");
	printf("ConvertFormat (GLISTR)\n");
#ifdef SW_VER
	printf("  Version %s\n", SW_VER);
#endif
#ifdef SW_REV
	printf("  Revision %s\n", SW_REV);
#endif
	printf("Copyright (c) 2014 University of Pennsylvania. All rights reserved.\n");
	printf("See http://www.cbica.upenn.edu/sbia/software/license.html or COPYING file.\n");
	printf("==========================================================================\n");
}
void usage()
{
	printf("\n");
	printf("Options:\n\n");
	printf("-i  (--input        ) [input_image_file]    : input image file (input)\n");
	printf("-f  (--format       ) [format_name]         : voxel format: float, double, byte, short, int, ... (input)\n");
	printf("-c  (--compress     ) [0 or 1]              : use compression: 0 or 1 (input)\n");
	printf("-o  (--output       ) [output_image_file]   : output image file (output)\n");
	printf("\n");
	printf("-h  (--help         )                       : print this help\n");
	printf("-u  (--usage        )                       : print this help\n");
	printf("-V  (--version      )                       : print version info\n");
	printf("\n\n");
	printf("Example:\n\n");
	printf("ConvertFormat -i [input_image_file] -f [format_name] -c 1 -o [output_image_file]\n");
}

int main(int argc, char* argv[])
{
	char input_image[1024] = {0,};
    char output_image[1024] = {0,};
	int nFormat = 0;
	bool bCompress = true;

	// parse command line
	{
		int i;
		if (argc == 1) {
			printf("use option -h or --help for help\n");
			exit(EXIT_FAILURE);
		}
		if (argc == 2) {
			if        (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
				version();
				usage();
				exit(EXIT_FAILURE);
			} else if (strcmp(argv[1], "-u") == 0 || strcmp(argv[1], "--usage") == 0) {
				version();
				usage();
				exit(EXIT_FAILURE);
			} else if (strcmp(argv[1], "-V") == 0 || strcmp(argv[1], "--version") == 0) {
				version();
				exit(EXIT_FAILURE);
			} else {
				printf("error: wrong arguments\n");
				printf("use option -h or --help for help\n");
				exit(EXIT_FAILURE);
			}
		}
		for (i = 1; i < argc; i++) {
			if (argv[i] == NULL || argv[i+1] == NULL) {
				printf("error: not specified argument\n");
				printf("use option -h or --help for help\n");
				exit(EXIT_FAILURE);
			}
			if        (strcmp(argv[i], "-i" ) == 0 || strcmp(argv[i], "--input"   ) == 0) { sprintf(input_image   , "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-f" ) == 0 || strcmp(argv[i], "--format"  ) == 0) {
				if (strcmp(argv[i+1], "float") == 0) {
					nFormat = 0;
				} else if (strcmp(argv[i+1], "double") == 0) {
					nFormat = 1;
				} else if (strcmp(argv[i+1], "byte") == 0) {
					nFormat = 2;
				} else if (strcmp(argv[i+1], "short") == 0) {
					nFormat = 3;
				} else if (strcmp(argv[i+1], "int") == 0) {
					nFormat = 4;
				} else {
					nFormat = 0;
				}
				i++;
			} else if (strcmp(argv[i], "-c" ) == 0 || strcmp(argv[i], "--compress") == 0) {
				if (atoi(argv[i+1]) == 0) {
					bCompress = false;
				} else {
					bCompress = true;
				}
				i++;
			} else if (strcmp(argv[i], "-o" ) == 0 || strcmp(argv[i], "--output"  ) == 0) { sprintf(output_image  , "%s", argv[i+1]); i++;
			} else {
				printf("error: %s is not recognized\n", argv[i]);
				printf("use option -h or --help for help\n");
				exit(EXIT_FAILURE);
			}
		}
		//
		if (input_image[0] == 0 || output_image[0] == 0) {
			printf("error: essential arguments are not specified\n");
			printf("use option -h or --help for help\n");
			exit(EXIT_FAILURE);
		}
	}
	
	// read fixed images
	if (nFormat == 0) {
		fImageReaderType::Pointer reader = fImageReaderType::New();
		fImageWriterType::Pointer writer = fImageWriterType::New();

		reader->SetFileName(input_image);
		try {
			reader->UpdateLargestPossibleRegion();
		} catch (itk::ExceptionObject& e) {
			std::cerr << e << std::endl;
			exit(EXIT_FAILURE);
		}

		writer->SetInput(reader->GetOutput());
		writer->SetFileName(output_image);
		writer->SetUseCompression(bCompress);
		try {
			writer->Update();
		} catch (itk::ExceptionObject& e) {
			std::cerr << e << std::endl;
			exit(EXIT_FAILURE);
		}
	} else if (nFormat == 1) {
		dImageReaderType::Pointer reader = dImageReaderType::New();
		dImageWriterType::Pointer writer = dImageWriterType::New();

		reader->SetFileName(input_image);
		try {
			reader->UpdateLargestPossibleRegion();
		} catch (itk::ExceptionObject& e) {
			std::cerr << e << std::endl;
			exit(EXIT_FAILURE);
		}

		writer->SetInput(reader->GetOutput());
		writer->SetFileName(output_image);
		writer->SetUseCompression(bCompress);
		try {
			writer->Update();
		} catch (itk::ExceptionObject& e) {
			std::cerr << e << std::endl;
			exit(EXIT_FAILURE);
		}
	} else if (nFormat == 2) {
		bImageReaderType::Pointer reader = bImageReaderType::New();
		bImageWriterType::Pointer writer = bImageWriterType::New();

		reader->SetFileName(input_image);
		try {
			reader->UpdateLargestPossibleRegion();
		} catch (itk::ExceptionObject& e) {
			std::cerr << e << std::endl;
			exit(EXIT_FAILURE);
		}

		writer->SetInput(reader->GetOutput());
		writer->SetFileName(output_image);
		writer->SetUseCompression(bCompress);
		try {
			writer->Update();
		} catch (itk::ExceptionObject& e) {
			std::cerr << e << std::endl;
			exit(EXIT_FAILURE);
		}
	} else if (nFormat == 3) {
		sImageReaderType::Pointer reader = sImageReaderType::New();
		sImageWriterType::Pointer writer = sImageWriterType::New();

		reader->SetFileName(input_image);
		try {
			reader->UpdateLargestPossibleRegion();
		} catch (itk::ExceptionObject& e) {
			std::cerr << e << std::endl;
			exit(EXIT_FAILURE);
		}

		writer->SetInput(reader->GetOutput());
		writer->SetFileName(output_image);
		writer->SetUseCompression(bCompress);
		try {
			writer->Update();
		} catch (itk::ExceptionObject& e) {
			std::cerr << e << std::endl;
			exit(EXIT_FAILURE);
		}
	} else if (nFormat == 4) {
		iImageReaderType::Pointer reader = iImageReaderType::New();
		iImageWriterType::Pointer writer = iImageWriterType::New();

		reader->SetFileName(input_image);
		try {
			reader->UpdateLargestPossibleRegion();
		} catch (itk::ExceptionObject& e) {
			std::cerr << e << std::endl;
			exit(EXIT_FAILURE);
		}

		writer->SetInput(reader->GetOutput());
		writer->SetFileName(output_image);
		writer->SetUseCompression(bCompress);
		try {
			writer->Update();
		} catch (itk::ExceptionObject& e) {
			std::cerr << e << std::endl;
			exit(EXIT_FAILURE);
		}
	} else {
		exit(EXIT_FAILURE);
	}
	
	exit(EXIT_SUCCESS);
}
