///////////////////////////////////////////////////////////////////////////////////////
// ReverseDeformationField.cxx
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
#if ITK_VERSION_MAJOR >= 4
#include <itkIterativeInverseDisplacementFieldImageFilter.h>
#else
#include <itkIterativeInverseDeformationFieldImageFilter.h>
//#include <itkInverseDeformationFieldImageFilter.h>
#endif
#include <itkCastImageFilter.h>
#include <itkImageFileWriter.h>


const unsigned int ImageDimension = 3;

typedef itk::Vector<float, ImageDimension> InputVectorType;
typedef float VectorComponentType;
typedef itk::Vector<VectorComponentType, ImageDimension> VectorType;

typedef itk::Image<InputVectorType, ImageDimension> InputDeformationFieldType;
typedef itk::Image<VectorType, ImageDimension> DeformationFieldType;

#if ITK_VERSION_MAJOR >= 4
typedef itk::IterativeInverseDisplacementFieldImageFilter<DeformationFieldType, DeformationFieldType> InvertFilterType;
#else
typedef itk::IterativeInverseDeformationFieldImageFilter<DeformationFieldType, DeformationFieldType> InvertFilterType;
//typedef itk::InverseDeformationFieldImageFilter<DeformationFieldType, DeformationFieldType> InvertFilterType;
#endif

typedef itk::CastImageFilter<InputDeformationFieldType, DeformationFieldType> D2FFieldCasterType;
typedef itk::CastImageFilter<DeformationFieldType, InputDeformationFieldType> F2DFieldCasterType;

typedef itk::ImageFileReader<InputDeformationFieldType> DeformationFieldReaderType;
typedef itk::ImageFileWriter<InputDeformationFieldType> DeformationFieldWriterType;


void version()
{
	printf("==========================================================================\n");
	printf("ReverseDeformationField (GLISTR)\n");
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
	printf("-i  (--input   ) [input_deformation_file]  : input deformation file (input)\n");
	printf("-o  (--output  ) [output_deformation_file] : output deformation file (output)\n");
	printf("-n  (--num_iter) [number_iterations]       : number of iterations (input)\n");
	printf("-s  (--stop_val) [stop_value]              : stop value (input)\n");
	printf("\n");
	printf("-h  (--help    )                           : print this help\n");
	printf("-u  (--usage   )                           : print this help\n");
	printf("-V  (--version )                           : print version info\n");
	printf("\n\n");
	printf("Example:\n\n");
	printf("ReverseDeformationField -i [input_deformation_file] -o [output_deformation_file] -n [iterations_number] -s [stop_value]\n");
}


int main(int argc, char* argv[])
{
#if 0
    bool ok = true;

    const char* input_deformation = NULL;
    const char* output_deformation = NULL;
	int NumberOfIterations;
	double StopValue;

	if (argc != 5) {
        usage();
        exit(EXIT_SUCCESS);
	}

	input_deformation = argv[1];
	output_deformation = argv[2];
	NumberOfIterations = atoi(argv[3]);
	StopValue = atof(argv[4]);

    if (input_deformation == NULL) {
        std::cerr << "Input deformation field not specified." << std::endl;
        ok = false;
    }
    if (output_deformation == NULL) {
        std::cerr << "Output deformation not specified." << std::endl;
        ok = false;
    }

    if (!ok) {
        std::cout << std::endl;
        usage();
        exit(EXIT_FAILURE);
    }
#else
	char input_deformation[1024] = {0,};
    char output_deformation[1024] = {0,};
	int NumberOfIterations = 0;
	double StopValue = 0;

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
			if        (strcmp(argv[i], "-i" ) == 0 || strcmp(argv[i], "--input"   ) == 0) { sprintf(input_deformation , "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-o" ) == 0 || strcmp(argv[i], "--output"  ) == 0) { sprintf(output_deformation, "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-n" ) == 0 || strcmp(argv[i], "--num_iter") == 0) { NumberOfIterations = atoi(argv[i+1]); i++;
			} else if (strcmp(argv[i], "-s" ) == 0 || strcmp(argv[i], "--stop_val") == 0) { StopValue = atof(argv[i+1]); i++;
			} else {
				printf("error: %s is not recognized\n", argv[i]);
				printf("use option -h or --help for help\n");
				exit(EXIT_FAILURE);
			}
		}
		//
		if (input_deformation[0] == 0 || output_deformation[0] == 0 || NumberOfIterations == 0 || StopValue == 0)
		{
			printf("error: essential arguments are not specified\n");
			printf("use option -h or --help for help\n");
			exit(EXIT_FAILURE);
		}
	}
#endif
  
    DeformationFieldReaderType::Pointer field_reader = DeformationFieldReaderType::New();
    field_reader->SetFileName(input_deformation);

    try {
        field_reader->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }
 
    D2FFieldCasterType::Pointer input_field_caster = D2FFieldCasterType::New();
    input_field_caster->SetInput(field_reader->GetOutput());

    InvertFilterType::Pointer filter = InvertFilterType::New();

    DeformationFieldType::Pointer field = DeformationFieldType::New();

    DeformationFieldType::RegionType region;
    region = field_reader->GetOutput()->GetLargestPossibleRegion();

    filter->SetInput(input_field_caster->GetOutput()); 
    filter->SetStopValue(StopValue);
    filter->SetNumberOfIterations(NumberOfIterations);

    try {
        filter->UpdateLargestPossibleRegion();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

    F2DFieldCasterType::Pointer output_field_caster = F2DFieldCasterType::New();
    output_field_caster->SetInput(filter->GetOutput());

    DeformationFieldWriterType::Pointer field_writer = DeformationFieldWriterType::New();

    field_writer->SetInput (output_field_caster->GetOutput());
    field_writer->SetFileName(output_deformation);
	field_writer->SetUseCompression(true);

    try {
        field_writer->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}
