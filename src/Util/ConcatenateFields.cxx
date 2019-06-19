///////////////////////////////////////////////////////////////////////////////////////
// ConcatenateFields.cxx
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014 University of Pennsylvania. All rights reserved.
// See http://www.cbica.upenn.edu/sbia/software/license.html or COYPING file.
//
// Contact: SBIA Group <sbia-software at uphs.upenn.edu>
///////////////////////////////////////////////////////////////////////////////////////

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include "itkDisplacementFieldCompositionFilter.h"


template <unsigned int Dimension>
void ConcatenateFields(char* FixToInterDef_file, char* InterToMovDef_file, char* FixToMovDef_file)
{
	typedef float PixelType;
	typedef itk::Vector< PixelType, Dimension > VectorPixelType;
	typedef itk::Image< VectorPixelType, Dimension > DeformationFieldType;
	typedef itk::ImageFileReader< DeformationFieldType > FieldReaderType;

	typename FieldReaderType::Pointer FixToInterFieldReader = FieldReaderType::New();
	FixToInterFieldReader->SetFileName(FixToInterDef_file);

	try
	{
		FixToInterFieldReader->Update();
	}
	catch( itk::ExceptionObject& err )
	{
		std::cout << "Could not read the field from fixed to intermediate." << std::endl;
		std::cout << err << std::endl;
		exit( EXIT_FAILURE );
	}

	typename FieldReaderType::Pointer InterToMovFieldReader = FieldReaderType::New();
	InterToMovFieldReader->SetFileName(InterToMovDef_file);

	try
	{
		InterToMovFieldReader->Update();
	}
	catch( itk::ExceptionObject& err )
	{
		std::cout << "Could not read the input field from intermediate to moving." << std::endl;
		std::cout << err << std::endl;
		exit( EXIT_FAILURE );
	}  

	typedef itk::DisplacementFieldCompositionFilter<DeformationFieldType,DeformationFieldType> ComposerType;
	typename ComposerType::Pointer composer = ComposerType::New();

	composer->SetInput(0, FixToInterFieldReader->GetOutput());
	composer->SetInput(1, InterToMovFieldReader->GetOutput());

	composer->Update();

	typedef itk::ImageFileWriter< DeformationFieldType > FieldWriterType;
	typename FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
	fieldWriter->SetFileName(FixToMovDef_file);
	fieldWriter->SetInput(composer->GetOutput());
	//fieldWriter->SetUseCompression(false);
	fieldWriter->SetUseCompression(true);

	try
	{
		fieldWriter->Update();
	}
	catch(itk::ExceptionObject& err)
	{
		std::cout << "Unexpected error." << std::endl;
		std::cout << err << std::endl;
		exit( EXIT_FAILURE );
	}    
}


void version()
{
	printf("==========================================================================\n");
	printf("ConcatenateFields (GLISTR)\n");
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
	printf("-fi (--fix_to_inter) [fix_to_inter_deformation_file] : fix to inter deformation file (input)\n");
	printf("-im (--inter_to_mov) [inter_to_mov_deformation_file] : inter to mov deformation file (input)\n");
	printf("-fm (--fix_to_mov  ) [fix_to_mov_deformation_file]   : fix to mov deformation file (output)\n");
	printf("\n");
	printf("-h  (--help        )                      : print this help\n");
	printf("-u  (--usage       )                      : print this help\n");
	printf("-V  (--version     )                      : print version info\n");
	printf("\n\n");
	printf("Example:\n\n");
	printf("ConcatenateFields -fi [fix_to_inter_deformation_file] -im [inter_to_mov_deformation_file] -fm [fix_to_mov_deformation_file]\n");
}


int main(int argc, char * argv[])
{
#if 0
	const char* FixToInterDef_file = NULL;
    const char* InterToMovDef_file = NULL;
	const char* FixToMovDef_file = NULL;

	if (argc < 4) { 
		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0] << " FixToInter-Def InterToMov-Def FixToMov-Def(Output) " << std::endl;
		return EXIT_FAILURE;
	}

	FixToInterDef_file = argv[1];
	InterToMovDef_file = argv[2];
	FixToMovDef_file = argv[3];
#else
	char FixToInterDef_file[1024] = {0,};
    char InterToMovDef_file[1024] = {0,};
	char FixToMovDef_file[1024] = {0,};

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
			if        (strcmp(argv[i], "-fi" ) == 0 || strcmp(argv[i], "--fix_to_inter") == 0) { sprintf(FixToInterDef_file, "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-im" ) == 0 || strcmp(argv[i], "--inter_to_mov") == 0) { sprintf(InterToMovDef_file, "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-fm" ) == 0 || strcmp(argv[i], "--fix_to_mov"  ) == 0) { sprintf(FixToMovDef_file  , "%s", argv[i+1]); i++;
			} else {
				printf("error: %s is not recognized\n", argv[i]);
				printf("use option -h or --help for help\n");
				exit(EXIT_FAILURE);
			}
		}
		//
		if (FixToInterDef_file[0] == 0 || InterToMovDef_file[0] == 0 || FixToMovDef_file[0] == 0)
		{
			printf("error: essential arguments are not specified\n");
			printf("use option -h or --help for help\n");
			exit(EXIT_FAILURE);
		}
	}
#endif

	// Get the image dimension
	itk::ImageIOBase::Pointer imageIO;
	try
	{
		imageIO = itk::ImageIOFactory::CreateImageIO(FixToInterDef_file, itk::ImageIOFactory::ReadMode);
		if (imageIO) {
			imageIO->SetFileName(FixToInterDef_file);
			imageIO->ReadImageInformation();
		} else {
			std::cout << "Could not read the FixToInter field information." << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	catch (itk::ExceptionObject& err)
	{
		std::cout << "Could not read the FixToInter image information." << std::endl;
		std::cout << err << std::endl;
		exit( EXIT_FAILURE );
	}

	switch (imageIO->GetNumberOfDimensions())
	{
	case 2:
		ConcatenateFields<2>(FixToInterDef_file, InterToMovDef_file, FixToMovDef_file);
		break;
	case 3:
		ConcatenateFields<3>(FixToInterDef_file, InterToMovDef_file, FixToMovDef_file);
		break;
	default:
		std::cout << "Unsuported dimension" << std::endl;
		exit(EXIT_FAILURE);
	}

	return EXIT_SUCCESS;
}
