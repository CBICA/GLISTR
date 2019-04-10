///////////////////////////////////////////////////////////////////////////////////////
// ResampleImage.cxx
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014 University of Pennsylvania. All rights reserved.
// See http://www.cbica.upenn.edu/sbia/software/license.html or COYPING file.
//
// Contact: SBIA Group <sbia-software at uphs.upenn.edu>
///////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNiftiImageIO.h>


const unsigned int ImageDimension = 3;
typedef float InputPixelType;
typedef float OutputPixelType;

typedef itk::Image<InputPixelType, ImageDimension> InputImageType; 
typedef itk::Image<OutputPixelType, ImageDimension> OutputImageType; 

typedef itk::ImageFileReader<InputImageType> ReaderType;
typedef itk::ImageFileWriter<OutputImageType> WriterType;
typedef itk::ResampleImageFilter<InputImageType, OutputImageType> ResampleImageFilterType;
typedef double CoordRepType;
typedef itk::LinearInterpolateImageFunction<InputImageType, CoordRepType> InterpolatorType;


void str_strip_file2(char* str_file_ext, char* str_ext)
{
	if (strrchr(str_file_ext, '.') == NULL) {
		strcpy(str_ext, str_file_ext);
	} else {
		strcpy(str_ext, strrchr(str_file_ext, '.')+1);
	}
}

void version()
{
	printf("==========================================================================\n");
	printf("ResampleImage (GLISTR)\n");
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
	printf("-i  (--input    ) [input_image_file]     : input image file (input)\n");
	printf("-o  (--output   ) [output_image_file]    : output image file (output)\n");
	printf("-r  (--reference) [reference_image_file] : reference image file (input)\n");
	printf("-x  (--ratio_x  ) [ratio_x]              : ratio in x axis (input)\n");
	printf("-y  (--ratio_y  ) [ratio_y]              : ratio in y axis (input)\n");
	printf("-z  (--ratio_z  ) [ratio_z]              : ratio in z axis (input)\n");
	printf("\n");
	printf("-h  (--help     )                        : print this help\n");
	printf("-u  (--usage    )                        : print this help\n");
	printf("-V  (--version  )                        : print version info\n");
	printf("\n\n");
	printf("Example:\n\n");
	printf("ResampleImage -i [input_image_file] -o [output_image_file] -r [reference_image_file] -x [ratio_x] -y [ratio_y] -z [ratio_z]\n");
}


int main(int argc, char* argv[])
{
#if 0
    bool ok = true;

    const char* input_image_file     = NULL;
    const char* output_image_file    = NULL;
    const char* reference_image_file = NULL;
	double ratio[3];

	if (argc != 7) {
        usage();
        exit(EXIT_SUCCESS);
	}

	input_image_file = argv[1];
	output_image_file = argv[2];
	reference_image_file = argv[3];
	ratio[0] = atof(argv[4]);
	ratio[1] = atof(argv[5]);
	ratio[2] = atof(argv[6]);

    if (input_image_file == NULL) {
        std::cerr << "input image not specified." << std::endl;
        ok = false;
    }
    if (output_image_file == NULL) {
        std::cerr << "output image not specified." << std::endl;
        ok = false;
    }
    if (reference_image_file == NULL) {
        std::cerr << "reference image not specified." << std::endl;
        ok = false;
    }

    if (!ok) {
        std::cout << std::endl;
        usage();
        exit(EXIT_FAILURE);
    }
#else
	char input_image_file[1024] = {0,};
    char output_image_file[1024] = {0,};
    char reference_image_file[1024] = {0,};
	double ratio[3] = {0,};

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
			if        (strcmp(argv[i], "-i" ) == 0 || strcmp(argv[i], "--input"    ) == 0) { sprintf(input_image_file    , "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-o" ) == 0 || strcmp(argv[i], "--output"   ) == 0) { sprintf(output_image_file   , "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-r" ) == 0 || strcmp(argv[i], "--reference") == 0) { sprintf(reference_image_file, "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-x" ) == 0 || strcmp(argv[i], "--ratio_x"  ) == 0) { ratio[0] = atof(argv[i+1]); i++;
			} else if (strcmp(argv[i], "-y" ) == 0 || strcmp(argv[i], "--ratio_y"  ) == 0) { ratio[1] = atof(argv[i+1]); i++;
			} else if (strcmp(argv[i], "-z" ) == 0 || strcmp(argv[i], "--ratio_z"  ) == 0) { ratio[2] = atof(argv[i+1]); i++;
			} else {
				printf("error: %s is not recognized\n", argv[i]);
				printf("use option -h or --help for help\n");
				exit(EXIT_FAILURE);
			}
		}
		//
		if (input_image_file[0] == 0 || output_image_file[0] == 0 || reference_image_file[0] == 0 || ratio[0] == 0 || ratio[1] == 0 || ratio[2] == 0)
		{
			printf("error: essential arguments are not specified\n");
			printf("use option -h or --help for help\n");
			exit(EXIT_FAILURE);
		}
	}
#endif

    // read input image
    ReaderType::Pointer image_reader = ReaderType::New();

    image_reader->SetFileName(input_image_file);

    try {
        image_reader->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

    InputImageType::Pointer input_image = image_reader->GetOutput();
    input_image->DisconnectPipeline();
    image_reader = NULL;

    InputImageType::SizeType      input_size      = input_image->GetLargestPossibleRegion().GetSize();
    InputImageType::SpacingType   input_spacing   = input_image->GetSpacing();
    InputImageType::PointType     input_origin    = input_image->GetOrigin();
    InputImageType::DirectionType input_direction = input_image->GetDirection();

    // read geometry of reference image
    OutputImageType::SizeType      output_size;
    OutputImageType::SpacingType   output_spacing;
    OutputImageType::PointType     output_origin;
    OutputImageType::DirectionType output_direction;

    itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();
    imageIO->SetFileName(reference_image_file);
    try {
        imageIO->ReadImageInformation();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<double> axis;
    for (unsigned int i = 0; i < ImageDimension; i++) {
        output_origin [i] = imageIO->GetOrigin(i);
        output_spacing[i] = imageIO->GetSpacing(i) / ratio[i];
        output_size   [i] = (int)(imageIO->GetDimensions(i) * ratio[i] + 0.5);
        axis = imageIO->GetDirection(i);
        for (unsigned int j = 0; j < ImageDimension; j++) {
            output_direction[j][i] = axis[j];
        }
    }

    // adjust image geometry of tumor density map written by ForwardSolverDiffusion
    //
    // Note that the tumor density map has origin (0,0,0) which, however, corresponds
    // to the origin of the upsampled atlas space, not the origin of the downsampled
    // label map which was the actual input to the ForwardSolverDiffusion program.
    input_origin    = output_origin;
    input_direction = output_direction;

    input_image->SetOrigin   (input_origin);
    input_image->SetDirection(input_direction);

    // resample image
    ResampleImageFilterType::Pointer resamp_filter = ResampleImageFilterType::New();

    resamp_filter->SetInput(input_image);
    resamp_filter->SetInterpolator(InterpolatorType::New());

    resamp_filter->SetSize           (output_size);
    resamp_filter->SetOutputSpacing  (output_spacing);
    resamp_filter->SetOutputOrigin   (output_origin);
    resamp_filter->SetOutputDirection(output_direction);

    try {
        resamp_filter->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

    OutputImageType::Pointer output_image = resamp_filter->GetOutput();
    output_image->DisconnectPipeline();
    resamp_filter = NULL;

    // write resampled image
    WriterType::Pointer image_writer = WriterType::New();

	char str_ext[1024];
	str_strip_file2((char*)output_image_file, str_ext);
    if (strcmp(str_ext, "hdr") == 0 || strcmp(str_ext, "img") == 0) {
        itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();
        image_writer->SetImageIO(imageIO);
    }
    image_writer->SetInput(output_image);
    image_writer->SetFileName(output_image_file);
	image_writer->SetUseCompression(true);

    try {
        image_writer->Update(); 
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}

