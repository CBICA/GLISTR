///////////////////////////////////////////////////////////////////////////////////////
// NormalizeImage.cpp
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
#define USE_N4
//
#ifdef USE_N4
#include "itkN4MRIBiasFieldCorrectionImageFilter.h"
#else
#include "itkN3MRIBiasFieldCorrectionImageFilter.h"
#endif
#include "itkBSplineControlPointImageFilter.h"
#include "itkExpImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkShrinkImageFilter.h"
//
#include "itkHistogramMatchingImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNiftiImageIO.h"


#if (!defined(WIN32) && !defined(WIN64)) || !defined(_DEBUG)
char SUSAN_PATH[1024];
#endif


template<class TFilter>
class CommandIterationUpdate : public itk::Command
{
public:
	typedef CommandIterationUpdate   Self;
	typedef itk::Command             Superclass;
	typedef itk::SmartPointer<Self>  Pointer;
	itkNewMacro( Self );
protected:
	CommandIterationUpdate() {};
public:

	void Execute(itk::Object *caller, const itk::EventObject & event)
	{
		Execute( (const itk::Object *) caller, event);
	}

	void Execute(const itk::Object * object, const itk::EventObject & event)
	{
		const TFilter * filter =
			dynamic_cast< const TFilter * >( object );
		if( typeid( event ) != typeid( itk::IterationEvent ) )
		{ return; }

		std::cout << "Iteration " << filter->GetElapsedIterations()
			<< " (of " << filter->GetMaximumNumberOfIterations() << ").  ";
		std::cout << " Current convergence value = "
			<< filter->GetCurrentConvergenceMeasurement()
			<< " (threshold = " << filter->GetConvergenceThreshold()
			<< ")" << std::endl;
	}
};

void version()
{
	printf("==========================================================================\n");
	printf("NormalizeImage (GLISTR)\n");
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
	printf("-o  (--output       ) [output_image_file]   : output image file (output)\n");
	printf("-t  (--template     ) [template_image_file] : template image file (input)\n");
	printf("-a  (--template_wm_a) [template_wm_range_a] : template wm range a, intensity in [a,b] is used for scaling\n");
	printf("-b  (--template_wm_b) [template_wm_range_b] : template wm range b, intensity in [a,b] is used for scaling\n");
	printf("-s  (--apply_susan  ) [1 or 0]              : 1 (default) if apply susan smoothing, 0 otherwise\n");
	printf("-sb (--susan_bt     ) [bt]                  : bt for susan\n");
	printf("-sd (--susan_dt     ) [dt]                  : dt for susan\n");
	printf("-n  (--apply_n3     ) [1 or 0]              : 1 (default) if apply n3 bias field correction, 0 otherwise\n");
	printf("-sh (--scale_hist   ) [1 or 0]              : 1 (default) if scale via histogram mathing, 0 scale using maximum value\n");
	printf("-sm (--scale_max    ) [max]                 : max value for scaling\n");
	printf("-w  (--apply_window ) [1 or 0]              : 1 (default) if apply windowing, 0 otherwise\n");
	printf("-wa (--window_a     ) [window_a]            : window lower percent (default: 0.01)\n");
	printf("-wb (--window_b     ) [window_b]            : window upper percent (default: 0.09)\n");
	printf("\n");
	printf("-h  (--help         )                       : print this help\n");
	printf("-u  (--usage        )                       : print this help\n");
	printf("-V  (--version      )                       : print version info\n");
	printf("\n\n");
	printf("Example:\n\n");
	printf("NormalizeImage -i [input_image_file] -o [output_image_file] -t [template_image_file] -a [template_wm_range_a] -b [template_wm_range_b] -s [1 or 0] -n [1 or 0]\n");
}

int main(int argc, char* argv[])
{
	char input_image[1024] = {0,};
    char output_image[1024] = {0,};
    char template_image[1024] = {0,};
	bool apply_susan = true;
	bool apply_n3 = true;
	float temp_a = 0;
	float temp_b = 10000;
	float bt = 80;
	float dt = 0;
	bool scale_hist = true;
	float scale_max = -1;
	bool apply_window = true;
	float window_a = 0.01;
	float window_b = 0.99;

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
			} else if (strcmp(argv[i], "-o" ) == 0 || strcmp(argv[i], "--output"  ) == 0) { sprintf(output_image  , "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-t" ) == 0 || strcmp(argv[i], "--template") == 0) { sprintf(template_image, "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-s" ) == 0 || strcmp(argv[i], "--apply_susan") == 0) {
				if (atoi(argv[i+1]) == 0) {
					apply_susan = false;
				} else {
					apply_susan = true;
				}
				i++;
			} else if (strcmp(argv[i], "-sb" ) == 0 || strcmp(argv[i], "--susan_bt") == 0) {
				bt = atof(argv[i+1]);
				i++;
			} else if (strcmp(argv[i], "-sd" ) == 0 || strcmp(argv[i], "--susan_dt") == 0) {
				dt = atof(argv[i+1]);
				i++;
			} else if (strcmp(argv[i], "-n" ) == 0 || strcmp(argv[i], "--apply_n3") == 0) {
				if (atoi(argv[i+1]) == 0) {
					apply_n3 = false;
				} else {
					apply_n3 = true;
				}
				i++;
			} else if (strcmp(argv[i], "-a" ) == 0 || strcmp(argv[i], "--template_wm_a") == 0) {
				temp_a = atof(argv[i+1]);
				i++;
			} else if (strcmp(argv[i], "-b" ) == 0 || strcmp(argv[i], "--template_wm_b") == 0) {
				temp_b = atof(argv[i+1]);
				i++;
			} else if (strcmp(argv[i], "-sh" ) == 0 || strcmp(argv[i], "--scale_hist") == 0) {
				if (atoi(argv[i+1]) == 0) {
					scale_hist = false;
				} else {
					scale_hist = true;
				}
				i++;
			} else if (strcmp(argv[i], "-sm" ) == 0 || strcmp(argv[i], "--scale_max") == 0) {
				scale_max = atof(argv[i+1]);
				i++;
			} else if (strcmp(argv[i], "-w" ) == 0 || strcmp(argv[i], "--apply_window") == 0) {
				if (atoi(argv[i+1]) == 0) {
					apply_window = false;
				} else {
					apply_window = true;
				}
				i++;
			} else if (strcmp(argv[i], "-wa" ) == 0 || strcmp(argv[i], "--window_a") == 0) {
				window_a = atof(argv[i+1]);
				i++;
			} else if (strcmp(argv[i], "-wb" ) == 0 || strcmp(argv[i], "--window_b") == 0) {
				window_b = atof(argv[i+1]);
				i++;
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
		if (scale_hist) {
			if (template_image[0] == 0) {
				printf("error: essential arguments are not specified\n");
				printf("use option -h or --help for help\n");
				exit(EXIT_FAILURE);
			}
			if (temp_a < 0 || temp_b > 2000 || temp_b < temp_a) {
				printf("wm range is invalid\n");
				exit(EXIT_FAILURE);
			}
		}
	}

#if (!defined(WIN32) && !defined(WIN64)) || !defined(_DEBUG)
	{
		char MODULE_PATH[1024];
#if 0
		str_strip_file(argv[0], MODULE_PATH);
#else
		if (!GetModulePath(MODULE_PATH)) {
			printf("GetModulePath failed\n");
			exit(EXIT_FAILURE);
		}
#endif
		//
#if defined(WIN32) || defined(WIN64)
		sprintf(SUSAN_PATH, "%ssusan.exe", MODULE_PATH);
#else
		sprintf(SUSAN_PATH, "susan");
#endif
	}
#endif

	char input_image_name[1024];
	{
		char str_tmp[1024];
		str_strip_ext(input_image, input_image_name);
		str_strip_ext(input_image_name, str_tmp);
		if (str_tmp[0] != 0) {
			strcpy(input_image_name, str_tmp);
		}
	}

	float scale;
	FVolume input, output;
	int i, j, k;

	if (apply_window) {
		char input_image_window[1024];

		sprintf(input_image_window, "%s_window.nii.gz", input_image_name);

		float min_val, max_val;
		int* hist;
		int hist_num, hist_sum, hist_acc, hist_a, hist_b;
		int a_val, b_val;

		input.load(input_image, 1);

		max_val = -1;
		min_val = FLT_MAX;
		for (k = 0; k < input.m_vd_z; k++) {
			for (j = 0; j < input.m_vd_y; j++) {
				for (i = 0; i < input.m_vd_x; i++) {
					float val = input.m_pData[k][j][i][0];
					if (val == 0) continue;

					if (val > max_val) {
						max_val = val;
					}
					if (val < min_val) {
						min_val = val;
					}
				}
			}
		}

		hist_num = (int)max_val + 1;
		hist = (int*)malloc(hist_num * sizeof(int));
		for (i = 0; i < hist_num; i++) {
			hist[i] = 0;
		}

		hist_sum = 0;
		for (k = 0; k < input.m_vd_z; k++) {
			for (j = 0; j < input.m_vd_y; j++) {
				for (i = 0; i < input.m_vd_x; i++) {
					float val = input.m_pData[k][j][i][0];
					if (val == 0) continue;

					hist[(int)val]++;
					hist_sum++;
				}
			}
		}

		a_val = 0;
		b_val = (int)max_val;
		hist_a = (int)(window_a * hist_sum);
		hist_b = (int)(window_b * hist_sum);

		hist_acc = 0;
		for (i = 1; i < hist_num; i++) {
			hist_acc += hist[i];
			if (hist_acc > hist_a) {
				a_val = i;
				break;
			}
		}
		hist_acc = 0;
		for (i = 1; i < hist_num; i++) {
			hist_acc += hist[i];
			if (hist_acc > hist_b) {
				b_val = i-1;
				break;
			}
		}

		for (k = 0; k < input.m_vd_z; k++) {
			for (j = 0; j < input.m_vd_y; j++) {
				for (i = 0; i < input.m_vd_x; i++) {
					float val = input.m_pData[k][j][i][0];
					if (val == 0) continue;

					if (val < a_val) {
						input.m_pData[k][j][i][0] = a_val;
					}
					if (val > b_val) {
						input.m_pData[k][j][i][0] = b_val;
					}
				}
			}
		}

		input.save(input_image_window, 1);
		if (!ChangeNIIHeader(input_image_window, input_image)) {
			TRACE("ChangeNIIHeader failed\n");
		}

		strcpy(input_image, input_image_window);

		free(hist);
	}

	if (apply_susan) {
		char input_image_susan[1024];
		char szCmdLine[2048];

		sprintf(input_image_susan, "%s_sus.nii.gz", input_image_name);

		putenv((char*)"FSLOUTPUTTYPE=NIFTI_GZ");
		//
		{
			//susan input.hdr 80 0 3D 1 0  output.hdr
			sprintf(szCmdLine, "%s %s %f %f 3D 1 0  %s",
				SUSAN_PATH, input_image, bt, dt, input_image_susan);
			TRACE("%s\n", szCmdLine);
			//
			if (!ExecuteProcess(szCmdLine)) {
				exit(EXIT_FAILURE);
			}
		}

		strcpy(input_image, input_image_susan);
	}

	if (apply_n3) {
		char input_image_n3[1024];
		const int ImageDimension = 3;
		//
		if (apply_susan) {
			sprintf(input_image_n3, "%s_sus_n3.nii.gz", input_image_name);
		} else {
			sprintf(input_image_n3, "%s_n3.nii.gz", input_image_name);
		}
		//
		typedef float RealType;

		typedef itk::Image<RealType, ImageDimension> ImageType;
		typedef itk::Image<unsigned char, ImageDimension> MaskImageType;

		typedef itk::ImageFileReader<ImageType> ReaderType;
		ReaderType::Pointer reader = ReaderType::New();
		//reader->SetFileName( argv[2] );
		reader->SetFileName(input_image);
		reader->Update();

		typedef itk::ShrinkImageFilter<ImageType, ImageType> ShrinkerType;
		ShrinkerType::Pointer shrinker = ShrinkerType::New();
		shrinker->SetInput( reader->GetOutput() );
		shrinker->SetShrinkFactors( 1 );

		MaskImageType::Pointer maskImage = NULL;

		/*
		if( argc > 5 )
		{
			typedef itk::ImageFileReader<MaskImageType> MaskReaderType;
			MaskReaderType::Pointer maskreader = MaskReaderType::New();
			maskreader->SetFileName( argv[5] );

			try
			{
				maskreader->Update();
				maskImage = maskreader->GetOutput();
			}
			catch(...)
			{
				std::cout << "Mask file not read.  Generating mask file using otsu"
					<< " thresholding." << std::endl;
			}
		}
		//*/
		if( !maskImage )
		{
			typedef itk::OtsuThresholdImageFilter<ImageType, MaskImageType>
				ThresholderType;
			ThresholderType::Pointer otsu = ThresholderType::New();
			otsu->SetInput( reader->GetOutput() );
			otsu->SetNumberOfHistogramBins( 200 );
			otsu->SetInsideValue( 0 );
			otsu->SetOutsideValue( 1 );
			otsu->Update();

			maskImage = otsu->GetOutput();
		}
		typedef itk::ShrinkImageFilter<MaskImageType, MaskImageType> MaskShrinkerType;
		MaskShrinkerType::Pointer maskshrinker = MaskShrinkerType::New();
		maskshrinker->SetInput( maskImage );
		maskshrinker->SetShrinkFactors( 1 );

		/*
		if( argc > 4 )
		{
			shrinker->SetShrinkFactors( atoi( argv[4] ) );
			maskshrinker->SetShrinkFactors( atoi( argv[4] ) );
		}
		//*/
		shrinker->Update();
		maskshrinker->Update();

#ifdef USE_N4
		typedef itk::N4MRIBiasFieldCorrectionImageFilter<ImageType, MaskImageType, ImageType> CorrecterType;
#else
		typedef itk::N3MRIBiasFieldCorrectionImageFilter<ImageType, MaskImageType, ImageType> CorrecterType;
#endif
		CorrecterType::Pointer correcter = CorrecterType::New();
		correcter->SetInput( shrinker->GetOutput() );
		correcter->SetMaskImage( maskshrinker->GetOutput() );

		/*
		if( argc > 6 )
		{
#ifdef USE_N4
			CorrecterType::VariableSizeArrayType maximumNumberOfIterations(1);
			maximumNumberOfIterations[0] = atoi(argv[6]);
			correcter->SetMaximumNumberOfIterations( maximumNumberOfIterations );
#else
			correcter->SetMaximumNumberOfIterations( atoi( argv[6] ) );
#endif
		}
		if( argc > 7 )
		{
			correcter->SetNumberOfFittingLevels( atoi( argv[7] ) );
		}
		//*/

		typedef CommandIterationUpdate<CorrecterType> CommandType;
		CommandType::Pointer observer = CommandType::New();
		correcter->AddObserver( itk::IterationEvent(), observer );

		try
		{
			correcter->Update();
		}
		catch(...)
		{
			std::cerr << "Exception caught." << std::endl;
			return EXIT_FAILURE;
		}

		//correcter->Print( std::cout, 3 );

		// Reconstruct the bias field at full image resolution.  Divide
		// the original input image by the bias field to get the final
		// corrected image.
		typedef itk::BSplineControlPointImageFilter<CorrecterType::BiasFieldControlPointLatticeType, CorrecterType::ScalarImageType> BSplinerType;
		BSplinerType::Pointer bspliner = BSplinerType::New();
		bspliner->SetInput( correcter->GetLogBiasFieldControlPointLattice() );
		bspliner->SetSplineOrder( correcter->GetSplineOrder() );
		bspliner->SetSize(
			reader->GetOutput()->GetLargestPossibleRegion().GetSize() );
		bspliner->SetOrigin( reader->GetOutput()->GetOrigin() );
		bspliner->SetDirection( reader->GetOutput()->GetDirection() );
		bspliner->SetSpacing( reader->GetOutput()->GetSpacing() );
		bspliner->Update();

		ImageType::Pointer logField = ImageType::New();
		logField->SetOrigin( bspliner->GetOutput()->GetOrigin() );
		logField->SetSpacing( bspliner->GetOutput()->GetSpacing() );
		logField->SetRegions(
			bspliner->GetOutput()->GetLargestPossibleRegion().GetSize() );
		logField->SetDirection( bspliner->GetOutput()->GetDirection() );
		logField->Allocate();

		itk::ImageRegionIterator<CorrecterType::ScalarImageType> ItB(bspliner->GetOutput(),	bspliner->GetOutput()->GetLargestPossibleRegion());
		itk::ImageRegionIterator<ImageType> ItF(logField, logField->GetLargestPossibleRegion());
		for( ItB.GoToBegin(), ItF.GoToBegin(); !ItB.IsAtEnd(); ++ItB, ++ItF )
		{
			ItF.Set( ItB.Get()[0] );
		}

		typedef itk::ExpImageFilter<ImageType, ImageType> ExpFilterType;
		ExpFilterType::Pointer expFilter = ExpFilterType::New();
		expFilter->SetInput( logField );
		expFilter->Update();

		typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DividerType;
		DividerType::Pointer divider = DividerType::New();
		divider->SetInput1( reader->GetOutput() );
		divider->SetInput2( expFilter->GetOutput() );
		divider->Update();

		typedef itk::ImageFileWriter<ImageType> WriterType;
		WriterType::Pointer writer = WriterType::New();
		//writer->SetFileName( argv[3] );
		writer->SetFileName(input_image_n3);		
		writer->SetInput( divider->GetOutput() );
		writer->Update();

		/*
		if( argc > 8 )
		{
			typedef itk::ImageFileWriter<ImageType> WriterType;
			WriterType::Pointer writer = WriterType::New();
			writer->SetFileName( argv[8] );
			writer->SetInput( expFilter->GetOutput() );
			writer->Update();
		}
		//*/

		strcpy(input_image, input_image_n3);
	}

	if (scale_hist) {
		// using histogram matching
		char input_image_hist[1024];
		//
		typedef float pixelType;
		typedef itk::Image<pixelType, 3> ImageType;
		typedef itk::ImageFileReader<ImageType> ReaderType;
		typedef itk::HistogramMatchingImageFilter<ImageType, ImageType> FilterType;
		typedef itk::ImageFileWriter<FilterType::OutputImageType> WriterType;
		int numQuantiles = 40;

		sprintf(input_image_hist, "%s_hist.nii.gz", input_image_name);

		FilterType::Pointer filter = FilterType::New();
		ReaderType::Pointer sourceReader = ReaderType::New();
		ReaderType::Pointer refReader = ReaderType::New();
  
		sourceReader->SetFileName(input_image);
		refReader->SetFileName(template_image);
  
		filter->SetSourceImage(sourceReader->GetOutput());
		filter->SetReferenceImage(refReader->GetOutput());
		filter->ThresholdAtMeanIntensityOn();
		filter->SetNumberOfMatchPoints(numQuantiles);
		//
		WriterType::Pointer writer = WriterType::New();
  
		writer->SetInput(filter->GetOutput());
		writer->SetImageIO(itk::NiftiImageIO::New());
		writer->SetFileName(input_image_hist);

		try
		{
			writer->Update();
		}
		catch(...)
		{
			exit(EXIT_FAILURE);
		}

		input.load(input_image, 1);
		output.load(input_image_hist, 1);

		// get scale histogram
		{
			#define MAX_SC 200
			int sc_hist[MAX_SC]; // sc_hist[0] => sc = 1/100, sc_hist[199] => sc = MAX_SC/100
			float sc, sum, val1, val2;
			int sc_idx, sc_idx_max, sc_hist_max;
			int num;

			for (i = 0; i < MAX_SC; i++) {
				sc_hist[i] = 0;
			}

			sum = 0;
			num = 0;
			for (k = 0; k < input.m_vd_z; k++) {
				for (j = 0; j < input.m_vd_y; j++) {
					for (i = 0; i < input.m_vd_x; i++) {
						val1 = input.m_pData[k][j][i][0];
						if (val1 == 0) {
							continue;
						}
						val2 = output.m_pData[k][j][i][0];
						//
						if (val2 < temp_a || val2 > temp_b) {
							continue;
						}
						//
						sc = val2 / val1;

						sc_idx = (int)(sc * 100) - 1;
						if (sc_idx >= 0 && sc_idx < MAX_SC) {
							sc_hist[sc_idx]++;
						}

						sum += sc;
						num++;
					}
				}
			}

			sc_hist_max = sc_hist[0];
			sc_idx_max = 0;
			for (i = 1; i < MAX_SC; i++) {
				if (sc_hist_max < sc_hist[i]) {
					sc_hist_max = sc_hist[i];
					sc_idx_max = i;
				}
			}

			scale = sum / num;
			TRACE("scale_avg = %f\n", sum / num);

			//scale = (float)(sc_idx_max+1) / 100;
			TRACE("scale_med = %f\n", (float)(sc_idx_max+1) / 100);
		}
	} else {
		input.load(input_image, 1);
		output.allocate(input.m_vd_x, input.m_vd_y, input.m_vd_z);

		if (scale_max < 0) {
			for (k = 0; k < input.m_vd_z; k++) {
				for (j = 0; j < input.m_vd_y; j++) {
					for (i = 0; i < input.m_vd_x; i++) {
						float val = input.m_pData[k][j][i][0];
						if (val > scale_max) {
							scale_max = val;
						}
					}
				}
			}
		}

		scale = 255.0 / scale_max;
	}

	{
		int i, j, k;

		// multiply scale to input
		{
			float val;

			//sprintf(szCmdLine, "%s %s -scale %f -o %s", C3D_PATH, input_image, scale, output_image);
			//sprintf(szCmdLine, "%s %s -scale %f -clip 0 255 -o %s", C3D_PATH, input_image, scale, output_image);
			for (k = 0; k < input.m_vd_z; k++) {
				for (j = 0; j < input.m_vd_y; j++) {
					for (i = 0; i < input.m_vd_x; i++) {
						val = input.m_pData[k][j][i][0] * scale;
						output.m_pData[k][j][i][0] = val;
					}
				}
			}
		}

		output.save(output_image, 1);
		if (!ChangeNIIHeader(output_image, input_image)) {
			TRACE("ChangeNIIHeader failed\n");
		}
	}
	
	exit(EXIT_SUCCESS);
}
