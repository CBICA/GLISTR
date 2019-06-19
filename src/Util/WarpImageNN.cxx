///////////////////////////////////////////////////////////////////////////////////////
// WarpImageNN.cxx
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014 University of Pennsylvania. All rights reserved.
// See http://www.cbica.upenn.edu/sbia/software/license.html or COYPING file.
//
// Contact: SBIA Group <sbia-software at uphs.upenn.edu>
///////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <itkVector.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkCastImageFilter.h>
#include <itkVectorCastImageFilter.h>
#include <itkWarpImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNiftiImageIO.h>
#include <itkImageFileWriter.h>


const unsigned int ImageDimension = 3;

typedef float InputPixelType;
typedef float OutputPixelType;
typedef double InternalPixelType;
typedef itk::Vector<float, ImageDimension> InputVectorType;

typedef itk::Image<InputPixelType, ImageDimension> InputImageType; 
typedef itk::Image<OutputPixelType, ImageDimension> OutputImageType; 
typedef itk::Image<InternalPixelType, ImageDimension> InternalImageType;
typedef itk::Image<InputVectorType, ImageDimension> DeformationFieldType;

typedef itk::ImageFileReader<DeformationFieldType> DeformationFieldReaderType;
typedef itk::ImageFileReader<InputImageType> InputImageReaderType;
typedef itk::CastImageFilter<InputImageType, InternalImageType> C2DCastFilterType;
typedef itk::CastImageFilter<InternalImageType, OutputImageType> D2CCastFilterType;
typedef itk::WarpImageFilter<InternalImageType,InternalImageType,DeformationFieldType> WarperType;
typedef WarperType::CoordRepType CoordRepType;
/*
typedef itk::LinearInterpolateImageFunction<InternalImageType, CoordRepType> InterpolatorType;
/*/
typedef itk::NearestNeighborInterpolateImageFunction<InternalImageType, CoordRepType> InterpolatorType;
//*/
typedef itk::ImageFileWriter<OutputImageType> OutputImageWriterType;


void version()
{
	printf("==========================================================================\n");
	printf("WarpImageNN (GLISTR)\n");
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
	printf("-i  (--input    ) [input_image_file]       : input (moving) image file (input)\n");
	printf("-r  (--reference) [reference_image_file]   : reference image file (input)\n");
	printf("-o  (--output   ) [output_image_file]      : output image file (output)\n");
	printf("-d  (--def_field) [deformation_field_file] : reference to input (moving) deformation field file (input)\n");
	printf("\n");
	printf("-h  (--help     )                           : print this help\n");
	printf("-u  (--usage    )                           : print this help\n");
	printf("-V  (--version  )                           : print version info\n");
	printf("\n\n");
	printf("Example:\n\n");
	printf("WarpImageNN -i [input_image_file] -r [reference_image_file] -o [output_image_file] -d [deformation_field_file]\n");
}


int main( int argc, char * argv[] )
{
#if 0
    bool ok = true;

    const char* input_image = NULL;
    const char* reference_image = NULL;
    const char* output_image = NULL;
    const char* deformation_field = NULL;

	if (argc != 5) {
        usage();
        exit(EXIT_SUCCESS);
	}

	input_image = argv[1];
	reference_image = argv[2];
	output_image = argv[3];
	deformation_field = argv[4];

    if (input_image == NULL) {
        std::cerr << "Input image not specified." << std::endl;
        ok = false;
    }
    if (reference_image == NULL) {
        std::cerr << "Reference image not specified." << std::endl;
        ok = false;
    }
    if (output_image == NULL) {
        std::cerr << "Output image not specified." << std::endl;
        ok = false;
    }
    if (deformation_field == NULL) {
        std::cerr << "Deformation field not specified." << std::endl;
        ok = false;
    }

    if (!ok) {
        std::cout << std::endl;
        usage();
        exit(EXIT_FAILURE);
    }
#else
	char input_image[1024] = {0,};
    char reference_image[1024] = {0,};
    char output_image[1024] = {0,};
    char deformation_field[1024] = {0,};

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
			if        (strcmp(argv[i], "-i" ) == 0 || strcmp(argv[i], "--input"    ) == 0) { sprintf(input_image      , "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-r" ) == 0 || strcmp(argv[i], "--reference") == 0) { sprintf(reference_image  , "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-o" ) == 0 || strcmp(argv[i], "--output"   ) == 0) { sprintf(output_image     , "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-d" ) == 0 || strcmp(argv[i], "--def_field") == 0) { sprintf(deformation_field, "%s", argv[i+1]); i++;
			} else {
				printf("error: %s is not recognized\n", argv[i]);
				printf("use option -h or --help for help\n");
				exit(EXIT_FAILURE);
			}
		}
		//
		if (input_image[0] == 0 || reference_image[0] == 0 || output_image[0] == 0 || deformation_field[0] == 0)
		{
			printf("error: essential arguments are not specified\n");
			printf("use option -h or --help for help\n");
			exit(EXIT_FAILURE);
		}
	}
#endif

    InputImageReaderType::Pointer input_image_reader = InputImageReaderType::New();
    input_image_reader->SetFileName(input_image);

    try {
        input_image_reader->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

    InputImageReaderType::Pointer reference_image_reader = InputImageReaderType::New();
    reference_image_reader->SetFileName(reference_image);

    try {
        reference_image_reader->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

    DeformationFieldReaderType::Pointer field_reader = DeformationFieldReaderType::New();
    field_reader->SetFileName(deformation_field);

    try {
        field_reader->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

    field_reader->GetOutput()->SetOrigin(reference_image_reader->GetOutput()->GetOrigin());
    field_reader->GetOutput()->SetSpacing(reference_image_reader->GetOutput()->GetSpacing());
    field_reader->GetOutput()->SetDirection(reference_image_reader->GetOutput()->GetDirection());

    C2DCastFilterType::Pointer c2d_caster = C2DCastFilterType::New();
    c2d_caster->SetInput(input_image_reader->GetOutput());

    try {
        c2d_caster->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

    c2d_caster->GetOutput()->SetSpacing(reference_image_reader->GetOutput()->GetSpacing());
    c2d_caster->GetOutput()->SetOrigin(reference_image_reader->GetOutput()->GetOrigin());
    c2d_caster->GetOutput()->SetDirection(reference_image_reader->GetOutput()->GetDirection());

    WarperType::Pointer warper = WarperType::New();
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    warper->SetInterpolator(interpolator);
    warper->SetInput(c2d_caster->GetOutput());
#if ITK_VERSION_MAJOR >= 4
    warper->SetDisplacementField(field_reader->GetOutput());
#else
    warper->SetDeformationField(field_reader->GetOutput());
#endif
    warper->SetOutputSpacing(reference_image_reader->GetOutput()->GetSpacing());
    warper->SetOutputOrigin(reference_image_reader->GetOutput()->GetOrigin());
    warper->SetOutputDirection(reference_image_reader->GetOutput()->GetDirection());

    try {
        warper->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

    D2CCastFilterType::Pointer d2c_caster = D2CCastFilterType::New();
    d2c_caster->SetInput(warper->GetOutput());

    try {
        d2c_caster->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

    OutputImageWriterType::Pointer output_image_writer = OutputImageWriterType::New();
    itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();
    output_image_writer->SetImageIO(imageIO);
    output_image_writer->SetFileName(output_image);
    output_image_writer->SetInput(d2c_caster->GetOutput());

    try {
        output_image_writer->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}
