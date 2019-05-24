///////////////////////////////////////////////////////////////////////////////////////
// VolRegOptITK.cpp
// Developed by Dongjin Kwon
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2011-2014 Dongjin Kwon
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
//
//#undef TRACE
//#define TRACE printf

#include "Volume.h"
#include "VolumeBP.h"

#include "VolRegOpt.h"

#include <iostream>

#include <omp.h>

#include "itkVector.h"
#include "itkImageFileReader.h"
#if ITK_VERSION_MAJOR >= 4
#include "itkIterativeInverseDisplacementFieldImageFilter.h"
#else
#include "itkIterativeInverseDeformationFieldImageFilter.h"
#endif
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkWarpImageFilter.h"
#include "itkGaussianOperator.h"
#include "itkVectorNeighborhoodOperatorImageFilter.h"
#include "itkPointSet.h"
#if ITK_VERSION_MAJOR >= 4
#include "itkBSplineControlPointImageFunction.h"
#endif
#include "itkBSplineScatteredDataPointSetToImageFilter.h"
#if ITK_VERSION_MAJOR >= 4
#include "itkExponentialDisplacementFieldImageFilter.h"
#else
#include "itkExponentialDeformationFieldImageFilter.h"
#endif
//
#include "fams.h"


#if (!defined(WIN32) && !defined(WIN64)) || !defined(_DEBUG)
extern char REVERSE_DEFORMATION_FIELD_PATH[1024];
extern char CONCATENATE_FIELD_PATH[1024];
#endif


#define MAX_LEVELS 12
//
#define NumberOfFixedChannels NumberOfImageChannels
#define NumberOfMovingChannels NumberOfPriorChannels
//
#define PI						3.1415926535897932384626433832795
#define PI2M4					39.478417604357434475337963999505
#define eps						1e-8
#define epss					1e-32
#define INITIAL_MEAN_VALUE		1e32
//
typedef vnl_matrix_fixed<double, NumberOfFixedChannels, NumberOfFixedChannels> VarianceType;
typedef std::vector<VarianceType> VarianceVectorType;
typedef vnl_vector_fixed<double, NumberOfFixedChannels> MeanType;
typedef std::vector<MeanType> MeanVectorType;
//
#define ImageDimension			3
//
typedef itk::Vector<double, ImageDimension> InputVectorType;
typedef float VectorComponentType;
typedef itk::Vector<VectorComponentType, ImageDimension> VectorType;

typedef itk::Image<InputVectorType, ImageDimension> InputDeformationFieldType;
typedef itk::Image<VectorType, ImageDimension> DeformationFieldType;

#if ITK_VERSION_MAJOR >= 4
typedef itk::IterativeInverseDisplacementFieldImageFilter<DeformationFieldType, DeformationFieldType> InvertFilterType;
#else
typedef itk::IterativeInverseDeformationFieldImageFilter<DeformationFieldType, DeformationFieldType> InvertFilterType;
#endif

typedef itk::CastImageFilter<InputDeformationFieldType, DeformationFieldType> D2FFieldCasterType;
typedef itk::CastImageFilter<DeformationFieldType, InputDeformationFieldType> F2DFieldCasterType;

typedef itk::ImageFileReader<InputDeformationFieldType> DeformationFieldReaderType;
typedef itk::ImageFileWriter<InputDeformationFieldType> DeformationFieldWriterType;

#if ITK_VERSION_MAJOR >= 4
typedef itk::ExponentialDisplacementFieldImageFilter< DeformationFieldType, DeformationFieldType > FieldExponentiatorType;
#else
typedef itk::ExponentialDeformationFieldImageFilter< DeformationFieldType, DeformationFieldType > FieldExponentiatorType;
#endif
typedef itk::WarpVectorImageFilter< DeformationFieldType, DeformationFieldType, DeformationFieldType > VectorWarperType;
typedef itk::VectorLinearInterpolateNearestNeighborExtrapolateImageFunction< DeformationFieldType, double > FieldInterpolatorType;
typedef itk::AddImageFilter< DeformationFieldType, DeformationFieldType, DeformationFieldType> AdderType;

extern BOOL ComputeFlow3D_TRWS_Decomposed(RVolume& dcv, int wsize, REALV label_sx, REALV label_sy, REALV label_sz, int lmode, REALV alpha_O1, REALV d_O1, REALV alpha_O2, REALV d_O2, REALV gamma, int nIterations,
	RVolume* xx, RVolume* yy, RVolume* zz, RVolume& vx, RVolume& vy, RVolume& vz, int vd_x, int vd_y, int vd_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV in_scv_w_O1F2, REALV in_scv_w_O2F2, REALV in_scv_w_O2F3, double* pEnergy = NULL, double* pLowerBound = NULL, RVolume* sc = NULL, int ffd = 0);

BOOL ReverseDeformationField(char* input_deformation, char* output_deformation)
{
    DeformationFieldReaderType::Pointer field_reader = DeformationFieldReaderType::New();
    field_reader->SetFileName(input_deformation);

    try {
        field_reader->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
		return FALSE;
    }
 
    D2FFieldCasterType::Pointer input_field_caster = D2FFieldCasterType::New();
    input_field_caster->SetInput(field_reader->GetOutput());

    InvertFilterType::Pointer filter = InvertFilterType::New();

    DeformationFieldType::Pointer field = DeformationFieldType::New();

    DeformationFieldType::RegionType region;
    region = field_reader->GetOutput()->GetLargestPossibleRegion();

    filter->SetInput(input_field_caster->GetOutput()); 
    filter->SetStopValue(0.1);

    try {
        filter->UpdateLargestPossibleRegion();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
		return FALSE;
    }

    F2DFieldCasterType::Pointer output_field_caster = F2DFieldCasterType::New();
    output_field_caster->SetInput(filter->GetOutput());

    DeformationFieldWriterType::Pointer field_writer = DeformationFieldWriterType::New();

    field_writer->SetInput (output_field_caster->GetOutput());
    field_writer->SetFileName(output_deformation);

    try {
        field_writer->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
		return FALSE;
    }

	return TRUE;
}

BOOL ReverseDeformationField(FVolume& vx, FVolume& vy, FVolume& vz, FVolume& vxr, FVolume& vyr, FVolume& vzr) {
	int vd_x, vd_y, vd_z;
	float vd_dx, vd_dy, vd_dz;
	float vd_ox, vd_oy, vd_oz;
	char v3_hdr_name[1024];
	char v3_img_name[1024];
	char v3r_hdr_name[1024];
	char v3r_img_name[1024];
	char szCmdLine[1024];

	sprintf(v3_hdr_name, "tmp_v3.mhd");
	sprintf(v3_img_name, "tmp_v3.zraw");
	sprintf(v3r_hdr_name, "tmp_v3r.mhd");
	sprintf(v3r_img_name, "tmp_v3r.zraw");

	vd_x = vx.m_vd_x; vd_y = vx.m_vd_y; vd_z = vx.m_vd_z;

	vxr.allocate(vd_x, vd_y, vd_z);
	vyr.allocate(vd_x, vd_y, vd_z);
	vzr.allocate(vd_x, vd_y, vd_z);

	if (!SaveMHDData(NULL, v3_hdr_name, vx.m_pData, vy.m_pData, vz.m_pData, vd_x, vd_y, vd_z, 1, 1, 1, 0, 0, 0)) {
		return FALSE;
	}

	sprintf(szCmdLine, "%s -i %s -o %s -n %d -s %e", REVERSE_DEFORMATION_FIELD_PATH, v3_hdr_name, v3r_hdr_name, 15, 1e-6);
	//sprintf(szCmdLine, "%s -i %s -o %s -n %d -s %e", REVERSE_DEFORMATION_FIELD_PATH, v3_hdr_name, v3r_hdr_name, 30, 1e-6);
	//
	TRACE2("%s\n", szCmdLine);
	//
	if (!ExecuteProcess(szCmdLine)) {
		return FALSE;
	}

	if (!LoadMHDData(NULL, v3r_hdr_name, &vxr.m_pData, &vyr.m_pData, &vzr.m_pData, vd_x, vd_y, vd_z, vd_dx, vd_dy, vd_dz, vd_ox, vd_oy, vd_oz)) {
		return FALSE;
	}

	DeleteFile(v3_hdr_name);
	DeleteFile(v3_img_name);
	DeleteFile(v3r_hdr_name);
	DeleteFile(v3r_img_name);

	return TRUE;
}

BOOL ConcatenateDeformationFields(FVolume& ax, FVolume& ay, FVolume& az, FVolume& bx, FVolume& by, FVolume& bz, FVolume& cx, FVolume& cy, FVolume& cz) {
	int vd_x, vd_y, vd_z;
	float vd_dx, vd_dy, vd_dz;
	float vd_ox, vd_oy, vd_oz;
	char a_v_hdr_name[1024];
	char a_v_img_name[1024];
	char b_v_hdr_name[1024];
	char b_v_img_name[1024];
	char c_v_hdr_name[1024];
	char c_v_img_name[1024];
	char szCmdLine[1024];

	cx.copy(ax);
	cy.copy(ay);
	cz.copy(az);

	sprintf(a_v_hdr_name, "tmp_a_v.mhd");
	sprintf(a_v_img_name, "tmp_a_v.zraw");
	sprintf(b_v_hdr_name, "tmp_b_v.mhd");
	sprintf(b_v_img_name, "tmp_b_v.zraw");
	sprintf(c_v_hdr_name, "tmp_c_v.mhd");
	sprintf(c_v_img_name, "tmp_c_v.zraw");

	vd_x = ax.m_vd_x; vd_y = ax.m_vd_y; vd_z = ax.m_vd_z;

	if (!SaveMHDData(NULL, a_v_hdr_name, ax.m_pData, ay.m_pData, az.m_pData, vd_x, vd_y, vd_z, 1, 1, 1, 0, 0, 0)) {
		return FALSE;
	}
	if (!SaveMHDData(NULL, b_v_hdr_name, bx.m_pData, by.m_pData, bz.m_pData, vd_x, vd_y, vd_z, 1, 1, 1, 0, 0, 0)) {
		return FALSE;
	}

	sprintf(szCmdLine, "%s -fi %s -im %s -fm %s", CONCATENATE_FIELD_PATH, a_v_hdr_name, b_v_hdr_name, c_v_hdr_name);
	//
	TRACE2("%s\n", szCmdLine);
	//
	if (!ExecuteProcess(szCmdLine)) {
		return FALSE;
	}

	if (!LoadMHDData(NULL, c_v_hdr_name, &cx.m_pData, &cy.m_pData, &cz.m_pData, vd_x, vd_y, vd_z, vd_dx, vd_dy, vd_dz, vd_ox, vd_oy, vd_oz)) {
		return FALSE;
	}

	DeleteFile(a_v_hdr_name);
	DeleteFile(a_v_img_name);
	DeleteFile(b_v_hdr_name);
	DeleteFile(b_v_img_name);
	DeleteFile(c_v_hdr_name);
	DeleteFile(c_v_img_name);

	return TRUE;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef itk::Image<float, ImageDimension> ImageType;
typedef ImageType::Pointer ImagePointer;
typedef itk::Image<VectorType, ImageDimension> DeformationFieldType;
typedef DeformationFieldType::Pointer DeformationFieldPointer;

void ComposeFields(DeformationFieldPointer fieldin, DeformationFieldPointer field, DeformationFieldPointer fieldout, float step)
{
	typedef itk::Point<float, 3> VPointType;
	typedef DeformationFieldType::PixelType VectorType;
	typedef itk::WarpImageFilter<ImageType,ImageType, DeformationFieldType> WarperType;
	typedef DeformationFieldType FieldType;  
	typedef itk::ImageRegionIteratorWithIndex<DeformationFieldType> FieldIterator; 
	typedef ImageType FloatImageType;
	typedef itk::ImageFileWriter<ImageType> writertype;
	typedef DeformationFieldType::IndexType IndexType;
	typedef DeformationFieldType::PointType PointType;
	typedef itk::VectorLinearInterpolateImageFunction<DeformationFieldType, float> DefaultInterpolatorType;
	VectorType zero;  
	ImageType::SpacingType oldspace;
	ImageType::SpacingType newspace;
	DefaultInterpolatorType::Pointer vinterp;
	VPointType pointIn1;
	VPointType pointIn2;
	DefaultInterpolatorType::ContinuousIndexType contind;
	VPointType pointIn3;
	IndexType index;
	//bool dosample;
	VectorType disp;
	DefaultInterpolatorType::OutputType disp2;
	VectorType out;
	int ct, jj;

	if (!fieldout) {
		fieldout=DeformationFieldType::New();
		fieldout->SetSpacing(fieldin->GetSpacing());
		fieldout->SetOrigin(fieldin->GetOrigin());
		fieldout->SetDirection(fieldin->GetDirection());
		fieldout->SetLargestPossibleRegion(fieldin->GetLargestPossibleRegion());
		fieldout->SetRequestedRegion(fieldin->GetLargestPossibleRegion());
		fieldout->SetBufferedRegion(fieldin->GetLargestPossibleRegion());
		fieldout->Allocate();
		zero.Fill(0);
		fieldout->FillBuffer(zero);
	}

	oldspace = field->GetSpacing();
	newspace = fieldin->GetSpacing();

	vinterp = DefaultInterpolatorType::New();
	vinterp->SetInputImage(field);

	ct = 0;
	FieldIterator m_FieldIter(fieldin, fieldin->GetLargestPossibleRegion());
	for (m_FieldIter.GoToBegin(); !m_FieldIter.IsAtEnd(); ++m_FieldIter) {
		index = m_FieldIter.GetIndex();
		//dosample = true;
		//if (dosample) {
			fieldin->TransformIndexToPhysicalPoint(index, pointIn1);
			disp = m_FieldIter.Get();
			for (jj = 0; jj < ImageDimension; jj++) {
				pointIn2[jj] = disp[jj] + pointIn1[jj];
			}
			if (vinterp->IsInsideBuffer(pointIn2)) {
				disp2 = vinterp->Evaluate(pointIn2);
			} else {
				disp2.Fill(0);
			}
			for (jj = 0; jj < ImageDimension; jj++) {
				pointIn3[jj] = disp2[jj]*step + pointIn2[jj];
			}

			for (jj = 0; jj < ImageDimension; jj++) {
				out[jj] = pointIn3[jj] - pointIn1[jj];
			}

			fieldout->SetPixel(m_FieldIter.GetIndex(), out);
			ct++;
		//}
	}
}

// use s <- s o exp(u)
void ComposeFieldsExp(DeformationFieldPointer fieldin, DeformationFieldPointer field, DeformationFieldPointer fieldout)
{
	typedef FieldExponentiatorType::Pointer FieldExponentiatorPointer;
	typedef VectorWarperType::Pointer VectorWarperPointer; 
	typedef FieldInterpolatorType::Pointer FieldInterpolatorPointer;
	typedef FieldInterpolatorType::OutputType FieldInterpolatorOutputType;
	typedef AdderType::Pointer AdderPointer;

	FieldExponentiatorPointer Exponentiator;
	VectorWarperPointer Warper;
	AdderPointer Adder;

	Exponentiator = FieldExponentiatorType::New();
	Warper = VectorWarperType::New();
	FieldInterpolatorPointer VectorInterpolator = FieldInterpolatorType::New();
	Warper->SetInterpolator(VectorInterpolator);

	Adder = AdderType::New();
	Adder->InPlaceOn();

	// compute the exponential
	Exponentiator->SetInput(field);

	//const double imposedMaxUpStep = this->GetMaximumUpdateStepLength();
	const double imposedMaxUpStep = 2.0;
	if (imposedMaxUpStep > 0.0) {
		// max(norm(Phi))/2^N <= 0.25*pixelspacing
		const double numiterfloat = 2.0 + vcl_log(imposedMaxUpStep)/vnl_math::ln2;
		unsigned int numiter = 0;
		if (numiterfloat > 0.0) {
			numiter = static_cast<unsigned int>(vcl_ceil(numiterfloat));
		}

		Exponentiator->AutomaticNumberOfIterationsOff();
		Exponentiator->SetMaximumNumberOfIterations(numiter);
	} else {
		Exponentiator->AutomaticNumberOfIterationsOn();
		// just set a high value so that automatic number of step
		// is not thresholded
		Exponentiator->SetMaximumNumberOfIterations(2000u);
	}

	Exponentiator->GetOutput()->SetRequestedRegion(fieldin->GetRequestedRegion());

	Exponentiator->Update();

	// compose the vector fields
	Warper->SetOutputOrigin(fieldin->GetOrigin());
	Warper->SetOutputSpacing(fieldin->GetSpacing());
	Warper->SetOutputDirection(fieldin->GetDirection());
	Warper->SetInput(fieldin);
#if ITK_VERSION_MAJOR >= 4
	Warper->SetDisplacementField(Exponentiator->GetOutput());
#else
	Warper->SetDeformationField(Exponentiator->GetOutput());
#endif

	Warper->Update();

	Adder->SetInput1(Warper->GetOutput());
	Adder->SetInput2(Exponentiator->GetOutput());

	Adder->GetOutput()->SetRequestedRegion(fieldin->GetRequestedRegion());

	// Triggers update
	Adder->Update();

	// Region passing stuff
	fieldout->Graft(Adder->GetOutput());
	fieldout->Modified();
}

float ReverseField(DeformationFieldPointer field, DeformationFieldPointer rField, float weight = 1.0, float toler = 0.1, int maxiter = 20)
{
	typedef DeformationFieldType::PixelType VectorType;
	typedef DeformationFieldType::IndexType IndexType;
	typedef VectorType::ValueType ScalarType;  
	typedef itk::ImageRegionIteratorWithIndex<DeformationFieldType> Iterator;
	typedef DeformationFieldType::SizeType SizeType;
	ImagePointer floatImage;
	DeformationFieldPointer lagrangianInitCond;
	DeformationFieldPointer eulerianInitCond;
	SizeType size;
	ImageType::SpacingType spacing;
	VectorType zero;
	IndexType index;
	VectorType update;
	float subpix;
	long long npix;
	float mag_max, mag, scale, val;
	float difmag, denergy, denergy2, laste, meandif, stepl, lastdifmag, epsilon;
	int ct, j, jj;
	
	zero.Fill(0);

	floatImage = ImageType::New();
	floatImage->SetLargestPossibleRegion(field->GetLargestPossibleRegion());
	floatImage->SetBufferedRegion(field->GetLargestPossibleRegion().GetSize());
	floatImage->SetSpacing(field->GetSpacing());
	floatImage->SetOrigin(field->GetOrigin());
	floatImage->SetDirection(field->GetDirection());
	floatImage->Allocate();

	lagrangianInitCond = DeformationFieldType::New();
	lagrangianInitCond->SetSpacing(field->GetSpacing());
	lagrangianInitCond->SetOrigin(field->GetOrigin());
	lagrangianInitCond->SetDirection(field->GetDirection());
	lagrangianInitCond->SetLargestPossibleRegion(field->GetLargestPossibleRegion());
	lagrangianInitCond->SetRequestedRegion(field->GetRequestedRegion());
	lagrangianInitCond->SetBufferedRegion(field->GetLargestPossibleRegion());
	lagrangianInitCond->Allocate();

	eulerianInitCond = DeformationFieldType::New();
	eulerianInitCond->SetSpacing(field->GetSpacing());
	eulerianInitCond->SetOrigin(field->GetOrigin());
	eulerianInitCond->SetDirection(field->GetDirection());
	eulerianInitCond->SetLargestPossibleRegion(field->GetLargestPossibleRegion());
	eulerianInitCond->SetRequestedRegion(field->GetRequestedRegion());
	eulerianInitCond->SetBufferedRegion(field->GetLargestPossibleRegion());
	eulerianInitCond->Allocate();

	size = field->GetLargestPossibleRegion().GetSize();
	spacing = field->GetSpacing();
	npix = 1;
	for (j = 0; j < ImageDimension; j++) {
		npix *= field->GetLargestPossibleRegion().GetSize()[j];
	}
	subpix = pow((float)ImageDimension, (float)ImageDimension)*0.5;

	{
		Iterator iter(field, field->GetLargestPossibleRegion());  
		mag_max = 0;
		for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter) {
			IndexType index = iter.GetIndex();
			VectorType vec1 = iter.Get();
			VectorType newvec = vec1 * weight;
			lagrangianInitCond->SetPixel(index, newvec);
			mag = 0;
			for (jj = 0; jj < ImageDimension; jj++) {
				mag += newvec[jj]*newvec[jj];
			}
			mag = sqrt(mag);
			if (mag > mag_max) {
				mag_max = mag;
			}
		}
	}

	eulerianInitCond->FillBuffer(zero);

	scale = (1.0) / mag_max;
	if (scale > 1.0) scale = 1.0;
	difmag = 10.0;
	ct = 0;
	denergy = 10;
	denergy2 = 10;
	laste = 1.0e9;
	meandif = 1.0e8;
	stepl = 2.0;
	lastdifmag = 0;
	epsilon = (float)size[0] / 256;
	if (epsilon > 1) epsilon = 1;

	{
		Iterator vfIter(rField, rField->GetLargestPossibleRegion());
		while (difmag > toler && ct < maxiter && meandif > 0.001) {
			denergy = laste - difmag;
			denergy2 = laste - meandif;
			laste = difmag;
			meandif = 0.0;

			ComposeFields(rField, lagrangianInitCond, eulerianInitCond, 1);

			difmag = 0.0;
			for (vfIter.GoToBegin(); !vfIter.IsAtEnd(); ++vfIter) {
				index = vfIter.GetIndex();
				update = eulerianInitCond->GetPixel(index); 
				mag = 0;
				for (j = 0; j < ImageDimension; j++) {
					update[j] *= (-1.0);
					mag += (update[j] / spacing[j]) * (update[j] / spacing[j]);
				}  
				mag = sqrt(mag);
				meandif += mag;
				if (mag > difmag) difmag = mag;

				eulerianInitCond->SetPixel(index, update); 
				floatImage->SetPixel(index, mag);
			}
			meandif /= (float)npix;     
			if (ct == 0) {
				epsilon = 0.75;
			} else {
				epsilon = 0.5;
			}
			stepl = difmag * epsilon;

			for (vfIter.GoToBegin(); !vfIter.IsAtEnd(); ++vfIter) {
				val = floatImage->GetPixel(vfIter.GetIndex());
				update = eulerianInitCond->GetPixel(vfIter.GetIndex());
				if (val > stepl) {
					update = update * (stepl/val);
				}
				VectorType upd = vfIter.Get() + update * (epsilon);
				vfIter.Set(upd);
			}

			ct++;
			lastdifmag = difmag;
		}
	}

	return difmag;
}

void SmoothField(DeformationFieldPointer field, float sig, unsigned int lodim = ImageDimension)
{
	typedef DeformationFieldType::PixelType VectorType;
	typedef VectorType::ValueType ScalarType;
	typedef itk::GaussianOperator<ScalarType,ImageDimension> OperatorType;
	typedef itk::VectorNeighborhoodOperatorImageFilter<DeformationFieldType, DeformationFieldType> SmootherType;
	typedef DeformationFieldType::PixelContainerPointer PixelContainerPointer;
	typedef itk::ImageRegionIteratorWithIndex<DeformationFieldType> Iterator;  
	DeformationFieldPointer tempField;
	OperatorType *oper;
	SmootherType::Pointer smoother;
	PixelContainerPointer swapPtr;
	ImageType::SpacingType spacing;
	DeformationFieldType::SizeType size;
	DeformationFieldType::IndexType index;
	float weight, weight2;
	int i, j;

	tempField = DeformationFieldType::New();
	tempField->SetSpacing(field->GetSpacing());
	tempField->SetOrigin(field->GetOrigin());
	tempField->SetDirection(field->GetDirection());
	tempField->SetLargestPossibleRegion(field->GetLargestPossibleRegion());
	tempField->SetRequestedRegion(field->GetRequestedRegion());
	tempField->SetBufferedRegion(field->GetBufferedRegion());
	tempField->Allocate();

	oper = new OperatorType;
	smoother = SmootherType::New();

	// graft the output field onto the mini-pipeline
	smoother->GraftOutput(tempField);

	spacing = field->GetSpacing();
	for (j = 0; j < (int)lodim; j++) {
		// smooth along this dimension
		oper->SetDirection(j);
		oper->SetVariance(sig);
		oper->SetMaximumError(0.001);
		oper->SetMaximumKernelWidth(256);
		oper->CreateDirectional();

		smoother->SetOperator(*oper);
		smoother->SetInput(field);
		smoother->Update();

		if (j < (int)lodim - 1) {
			// swap the containers
			swapPtr = smoother->GetOutput()->GetPixelContainer();
			smoother->GraftOutput(field);
			field->SetPixelContainer(swapPtr);
			smoother->Modified();
		}
	}

	// graft the output back to this filter
	tempField->SetPixelContainer(field->GetPixelContainer());

	//make sure boundary does not move
	weight = 1.0;
	if (sig < 0.5) {
		weight = 1.0 - 1.0*(sig/0.5);
	}
	weight2 = 1.0 - weight;

	size = field->GetLargestPossibleRegion().GetSize();
	Iterator outIter(field, field->GetLargestPossibleRegion());
	for (outIter.GoToBegin(); !outIter.IsAtEnd(); ++outIter) {
		bool onboundary = false;
		index = outIter.GetIndex();
		for (i = 0; i < ImageDimension;i++) {
			if (index[i] < 1 || index[i] >= (int)size[i] - 1) onboundary = true;
		}
		if (onboundary) {
			VectorType vec;
			vec.Fill(0.0);
			outIter.Set(vec);
		} else {
			VectorType svec=smoother->GetOutput()->GetPixel(index);
			outIter.Set(svec*weight+outIter.Get()*weight2);
		}
	}

	delete oper;
}

BOOL ComposeFields(FVolume& ax, FVolume& ay, FVolume& az, FVolume& bx, FVolume& by, FVolume& bz, FVolume& cx, FVolume& cy, FVolume& cz, float step, bool bUseExp = false)
{
	ImageType::SizeType size;
	ImageType::IndexType start;
	ImageType::RegionType region;
	ImageType::SpacingType spacing;
	double origin[ImageDimension];
    VectorType zero;
    zero.Fill(0);

	cx.allocate(ax.m_vd_x, ax.m_vd_y, ax.m_vd_z, 1, ax.m_vd_dx, ax.m_vd_dy, ax.m_vd_dz);
	cy.allocate(ax.m_vd_x, ax.m_vd_y, ax.m_vd_z, 1, ax.m_vd_dx, ax.m_vd_dy, ax.m_vd_dz);
	cz.allocate(ax.m_vd_x, ax.m_vd_y, ax.m_vd_z, 1, ax.m_vd_dx, ax.m_vd_dy, ax.m_vd_dz);

	size[0] = ax.m_vd_x;
	size[1] = ax.m_vd_y;
	size[2] = ax.m_vd_z;
	start[0] = 0;
	start[1] = 0;
	start[2] = 0;
	region.SetSize(size);
	region.SetIndex(start);
	//
	spacing[0] = ax.m_vd_dx;
	spacing[1] = ax.m_vd_dy;
	spacing[2] = ax.m_vd_dz;
	origin[0] = 0.0;
	origin[1] = 0.0;
	origin[2] = 0.0;

	DeformationFieldPointer fieldin;
	DeformationFieldPointer field;
	DeformationFieldPointer fieldout;

    fieldin = DeformationFieldType::New();
    fieldin->SetRegions(region);
    fieldin->SetSpacing(spacing);
	fieldin->SetOrigin(origin);
    fieldin->Allocate();
    fieldin->FillBuffer(zero);

    field = DeformationFieldType::New();
    field->SetRegions(region);
    field->SetSpacing(spacing);
	field->SetOrigin(origin);
    field->Allocate();
    field->FillBuffer(zero);

	fieldout = DeformationFieldType::New();
    fieldout->SetRegions(region);
    fieldout->SetSpacing(spacing);
	fieldout->SetOrigin(origin);
    fieldout->Allocate();
    fieldout->FillBuffer(zero);

	itk::ImageRegionIteratorWithIndex<DeformationFieldType> a_iter(fieldin, fieldin->GetLargestPossibleRegion());
	for (a_iter.GoToBegin(); !a_iter.IsAtEnd(); ++a_iter)
	{
		DeformationFieldType::IndexType index = a_iter.GetIndex();
		VectorType vec;

		vec[0] = ax.m_pData[index[2]][index[1]][index[0]][0];
		vec[1] = ay.m_pData[index[2]][index[1]][index[0]][0];
		vec[2] = az.m_pData[index[2]][index[1]][index[0]][0];

		a_iter.Set(vec);
	}

	itk::ImageRegionIteratorWithIndex<DeformationFieldType> b_iter(field, field->GetLargestPossibleRegion());
	for (b_iter.GoToBegin(); !b_iter.IsAtEnd(); ++b_iter)
	{
		DeformationFieldType::IndexType index = b_iter.GetIndex();
		VectorType vec;

		vec[0] = bx.m_pData[index[2]][index[1]][index[0]][0];
		vec[1] = by.m_pData[index[2]][index[1]][index[0]][0];
		vec[2] = bz.m_pData[index[2]][index[1]][index[0]][0];

		b_iter.Set(vec);
	}

	if (bUseExp) {
		ComposeFieldsExp(fieldin, field, fieldout);
	} else {
		ComposeFields(fieldin, field, fieldout, step);
	}

	itk::ImageRegionIteratorWithIndex<DeformationFieldType> c_iter(fieldout, fieldout->GetLargestPossibleRegion());
	for (c_iter.GoToBegin(); !c_iter.IsAtEnd(); ++c_iter)
	{
		DeformationFieldType::IndexType index = c_iter.GetIndex();
		VectorType vec;

		vec = c_iter.Get();

		cx.m_pData[index[2]][index[1]][index[0]][0] = vec[0];
		cy.m_pData[index[2]][index[1]][index[0]][0] = vec[1];
		cz.m_pData[index[2]][index[1]][index[0]][0] = vec[2];
	}

	return TRUE;
}

// caution: this function modifies the reverse field incrementally
BOOL ReverseField(FVolume& vx, FVolume& vy, FVolume& vz, FVolume& vxr, FVolume& vyr, FVolume& vzr)
{
	ImageType::SizeType size;
	ImageType::IndexType start;
	ImageType::RegionType region;
	ImageType::SpacingType spacing;
	double origin[ImageDimension];
    VectorType zero;
    zero.Fill(0);

	size[0] = vx.m_vd_x;
	size[1] = vx.m_vd_y;
	size[2] = vx.m_vd_z;
	start[0] = 0;
	start[1] = 0;
	start[2] = 0;
	region.SetSize(size);
	region.SetIndex(start);
	//
	spacing[0] = vx.m_vd_dx;
	spacing[1] = vx.m_vd_dy;
	spacing[2] = vx.m_vd_dz;
	origin[0] = 0.0;
	origin[1] = 0.0;
	origin[2] = 0.0;
	
	DeformationFieldPointer field;
	DeformationFieldPointer rField;

    field = DeformationFieldType::New();
    field->SetRegions(region);
    field->SetSpacing(spacing);
	field->SetOrigin(origin);
    field->Allocate();
    field->FillBuffer(zero);

    rField = DeformationFieldType::New();
    rField->SetRegions(region);
    rField->SetSpacing(spacing);
	rField->SetOrigin(origin);
    rField->Allocate();
    rField->FillBuffer(zero);

	itk::ImageRegionIteratorWithIndex<DeformationFieldType> f_iter(field, field->GetLargestPossibleRegion());
	for (f_iter.GoToBegin(); !f_iter.IsAtEnd(); ++f_iter)
	{
		DeformationFieldType::IndexType index = f_iter.GetIndex();
		VectorType vec;

		vec[0] = vx.m_pData[index[2]][index[1]][index[0]][0];
		vec[1] = vy.m_pData[index[2]][index[1]][index[0]][0];
		vec[2] = vz.m_pData[index[2]][index[1]][index[0]][0];

		f_iter.Set(vec);
	}

	itk::ImageRegionIteratorWithIndex<DeformationFieldType> f_inv_iter(rField, rField->GetLargestPossibleRegion());
	for (f_inv_iter.GoToBegin(); !f_inv_iter.IsAtEnd(); ++f_inv_iter)
	{
		DeformationFieldType::IndexType index = f_inv_iter.GetIndex();
		VectorType vec;

		vec[0] = vxr.m_pData[index[2]][index[1]][index[0]][0];
		vec[1] = vyr.m_pData[index[2]][index[1]][index[0]][0];
		vec[2] = vzr.m_pData[index[2]][index[1]][index[0]][0];

		f_inv_iter.Set(vec);
	}

	ReverseField(field, rField);

	for (f_inv_iter.GoToBegin(); !f_inv_iter.IsAtEnd(); ++f_inv_iter)
	{
		DeformationFieldType::IndexType index = f_inv_iter.GetIndex();
		VectorType vec;

		vec = f_inv_iter.Get();

		vxr.m_pData[index[2]][index[1]][index[0]][0] = vec[0];
		vyr.m_pData[index[2]][index[1]][index[0]][0] = vec[1];
		vzr.m_pData[index[2]][index[1]][index[0]][0] = vec[2];
	}

	return TRUE;
}

BOOL SmoothField(FVolume& vx, FVolume& vy, FVolume& vz, FVolume& vx_g, FVolume& vy_g, FVolume& vz_g, float sig)
{
	ImageType::SizeType size;
	ImageType::IndexType start;
	ImageType::RegionType region;
	ImageType::SpacingType spacing;
	double origin[ImageDimension];
    VectorType zero;
    zero.Fill(0);

	size[0] = vx.m_vd_x;
	size[1] = vx.m_vd_y;
	size[2] = vx.m_vd_z;
	start[0] = 0;
	start[1] = 0;
	start[2] = 0;
	region.SetSize(size);
	region.SetIndex(start);
	//
	spacing[0] = vx.m_vd_dx;
	spacing[1] = vx.m_vd_dy;
	spacing[2] = vx.m_vd_dz;
	origin[0] = 0.0;
	origin[1] = 0.0;
	origin[2] = 0.0;
	
	DeformationFieldPointer field;

    field = DeformationFieldType::New();
    field->SetRegions(region);
    field->SetSpacing(spacing);
	field->SetOrigin(origin);
    field->Allocate();
    field->FillBuffer(zero);
	
	itk::ImageRegionIteratorWithIndex<DeformationFieldType> f_iter(field, field->GetLargestPossibleRegion());
	for (f_iter.GoToBegin(); !f_iter.IsAtEnd(); ++f_iter)
	{
		DeformationFieldType::IndexType index = f_iter.GetIndex();
		VectorType vec;

		vec[0] = vx.m_pData[index[2]][index[1]][index[0]][0];
		vec[1] = vy.m_pData[index[2]][index[1]][index[0]][0];
		vec[2] = vz.m_pData[index[2]][index[1]][index[0]][0];

		f_iter.Set(vec);
	}

	SmoothField(field, sig);

	itk::ImageRegionIteratorWithIndex<DeformationFieldType> f_iter2(field, field->GetLargestPossibleRegion());
	for (f_iter2.GoToBegin(); !f_iter2.IsAtEnd(); ++f_iter2)
	{
		DeformationFieldType::IndexType index = f_iter2.GetIndex();
		VectorType vec;

		vec = f_iter2.Get();

		vx_g.m_pData[index[2]][index[1]][index[0]][0] = vec[0];
		vy_g.m_pData[index[2]][index[1]][index[0]][0] = vec[1];
		vz_g.m_pData[index[2]][index[1]][index[0]][0] = vec[2];
	}

	return TRUE;
}

double GetEnergy_CC(FVolume& vd1, FVolume& vd2, FVolume* mask1, FVolume* mask2, int dc_skip_back = 1, float dc_back_color = 0, int radius = 4) 
{
	int i, j, k, l, m, n;
	int vd1_x, vd1_y, vd1_z, vd2_x, vd2_y, vd2_z;
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	int ninv_size;
	int lx, ly, lz, rx, ry, rz;
	FVolume vd1t, vd2t;
	FVolume mask1t, mask2t;
	int vdt_x, vdt_y, vdt_z;
	//float xc, yc, zc;
	int ixc, iyc, izc;
	int ix, iy, iz;
	float vd1_val, vd2_val;
	float fixedMean, movingMean;
	float suma2, sumb2, suma, sumb, sumab, count;
	double sff, smm, sfm;
	double energy = 0;
#ifdef USE_CC_NCC
	double energy_cc = 0;
#endif
	//
	double localCrossCorrelation;

	vd1_x = vd1.m_vd_x;
	vd1_y = vd1.m_vd_y;
	vd1_z = vd1.m_vd_z;
	vd2_x = vd2.m_vd_x;
	vd2_y = vd2.m_vd_y;
	vd2_z = vd2.m_vd_z;

	if (vd1_x != vd2_x || vd1_y != vd2_y || vd1_z != vd2_z) {
		return 0;
	}

	ninv_x = 2 * radius + 1;
	ninv_y = 2 * radius + 1;
	ninv_z = 2 * radius + 1;
	ninv_size = ninv_x * ninv_y * ninv_z;
	//
	ninv_cx = ninv_x / 2;
	ninv_cy = ninv_y / 2;
	ninv_cz = ninv_z / 2;

	lx = ninv_cx;
	rx = ninv_cx+1;
	ly = ninv_cy;
	ry = ninv_cy+1;
	lz = ninv_cz;
	rz = ninv_cz+1;

	vdt_x = max(vd1_x+lx+rx, vd2_x+lx+rx);
	vdt_y = max(vd1_y+ly+ry, vd2_y+ly+ry);
	vdt_z = max(vd1_z+lz+rz, vd2_z+lz+rz);

	vd1t.allocate(vdt_x, vdt_y, vdt_z);
	vd2t.allocate(vdt_x, vdt_y, vdt_z);
	mask1t.allocate(vdt_x, vdt_y, vdt_z);
	mask2t.allocate(vdt_x, vdt_y, vdt_z);
	
	// translate images
	#pragma omp parallel for private(i,j,k,ixc,iyc,izc)
	for (k = 0; k < vdt_z; k++) {
		for (j = 0; j < vdt_y; j++) {
			for (i = 0; i < vdt_x; i++) {
				ixc = i - lx;
				iyc = j - ly;
				izc = k - lz;
				//if ((ixc <= 0) || (ixc >= vd1_x-1) || (iyc <= 0) || (iyc >= vd1_y-1) || (izc <= 0) || (izc >= vd1_z-1)) {
				if ((ixc < 0) || (ixc > vd1_x-1) || (iyc < 0) || (iyc > vd1_y-1) || (izc < 0) || (izc > vd1_z-1)) {
					vd1t.m_pData[k][j][i][0] = 0;
					vd2t.m_pData[k][j][i][0] = 0;
					//
					mask1t.m_pData[k][j][i][0] = 0;
					mask2t.m_pData[k][j][i][0] = 0;
				} else {
					vd1t.m_pData[k][j][i][0] = vd1.m_pData[izc][iyc][ixc][0];
					vd2t.m_pData[k][j][i][0] = vd2.m_pData[izc][iyc][ixc][0];
					//
					if (dc_skip_back == 1) {
						if (vd1.m_pData[izc][iyc][ixc][0] <= dc_back_color || vd2.m_pData[izc][iyc][ixc][0] <= dc_back_color) {
							mask1t.m_pData[k][j][i][0] = 0;
							mask2t.m_pData[k][j][i][0] = 0;
							continue;
						}
					}
					//
					if (mask1 != NULL) {
						mask1t.m_pData[k][j][i][0] = mask1->m_pData[izc][iyc][ixc][0];
					} else {
						mask1t.m_pData[k][j][i][0] = 1;
					}
					if (mask2 != NULL) {
						mask2t.m_pData[k][j][i][0] = mask2->m_pData[izc][iyc][ixc][0];
					} else {
						mask2t.m_pData[k][j][i][0] = 1;
					}
				}
			}
		}
	}

#ifdef USE_CC_NCC
	#pragma omp parallel for private(l,m,n,ixc,iyc,izc,count,suma,suma2,sumb,sumb2,sumab,i,j,k,ix,iy,iz,vd1_val,vd2_val,fixedMean,movingMean,sff,smm,sfm,localCrossCorrelation) shared(energy,energy_cc)
#else
	#pragma omp parallel for private(l,m,n,ixc,iyc,izc,count,suma,suma2,sumb,sumb2,sumab,i,j,k,ix,iy,iz,vd1_val,vd2_val,fixedMean,movingMean,sff,smm,sfm,localCrossCorrelation) shared(energy)
#endif
	/*
	for (n = 0; n < vd1_z; n++) {
		for (m = 0; m < vd1_y; m++) {
			for (l = 0; l < vd1_x; l++) {
	/*/
	for (n = radius; n < vd1_z-radius; n++) {
		for (m = radius; m < vd1_y-radius; m++) {
			for (l = radius; l < vd1_x-radius; l++) {
	//*/
				ixc = (int)(l - ninv_cx + lx);
				iyc = (int)(m - ninv_cy + ly);
				izc = (int)(n - ninv_cz + lz);
				//
				count = 0;
				suma2 = suma = 0;
				sumb2 = sumb = 0;
				sumab = 0;
				for (k = 0; k < ninv_z; k++) {
					for (j = 0; j < ninv_y; j++) {
						for (i = 0; i < ninv_x; i++) {
							ix = ixc + i;
							iy = iyc + j;
							iz = izc + k;
							//
							if (mask1t.m_pData[iz][iy][ix][0] > 0 && mask2t.m_pData[iz][iy][ix][0] > 0) {
								vd1_val = vd1t.m_pData[iz][iy][ix][0];
								vd2_val = vd2t.m_pData[iz][iy][ix][0];
								//
								suma  += vd1_val;
								suma2 += vd1_val * vd1_val;
								sumb  += vd2_val;
								sumb2 += vd2_val * vd2_val;
								sumab += vd1_val * vd2_val;
								count += 1;
							}
						}
					}
				}
				//
				ixc = l + lx;
				iyc = m + ly;
				izc = n + lz;
				//
				if (count > 0) {
					fixedMean  = suma / count;
					movingMean = sumb / count;
					sff = suma2 -  fixedMean*suma -  fixedMean*suma + count* fixedMean* fixedMean;
					smm = sumb2 - movingMean*sumb - movingMean*sumb + count*movingMean*movingMean;
					sfm = sumab - movingMean*suma -  fixedMean*sumb + count*movingMean* fixedMean;
					//
					if ((sff > 0) && (smm > 0)) {
					//if ((sff > 1e-1) && (smm > 1e-1)) {
					//if (sff*smm > 1.e-5) {
						if (sff*smm > 1.e-5) {
#ifdef USE_CC_NCC
							localCrossCorrelation = sfm / sqrt(sff * smm);
							if (localCrossCorrelation <= 1.0) {
								#pragma omp atomic
								energy += 1.0 - localCrossCorrelation;
							}
							//
							localCrossCorrelation = sfm*sfm / (sff * smm);
							if (localCrossCorrelation <= 1.0) {
								#pragma omp atomic
								energy_cc -= localCrossCorrelation;
							}
#else
							localCrossCorrelation = sfm*sfm / (sff * smm);
							if (localCrossCorrelation <= 1.0) {
								#pragma omp atomic
								energy -= localCrossCorrelation;
							}
#endif
						}
					}
				}
			}
		}
	}

#ifdef USE_CC_NCC
	TRACE2("energy_cc = %f\n", energy_cc);
#endif

	vd1t.clear();
	vd2t.clear();
	mask1t.clear();
	mask2t.clear();

	return energy;
}

//#define TEST_GET_VELOCITY_CC
double GetVelocity_CC(FVolume& vd1, FVolume& vd2, FVolume* mask1, FVolume* mask2, FVolume* vx, FVolume* vy, FVolume* vz, FVolume* vx_inv, FVolume* vy_inv, FVolume* vz_inv,
	int dc_skip_back = 1, float dc_back_color = 0, float weight = 1.0f, int radius = 4, float sigma = 3.0f) 
{
	int i, j, k, l, m, n;
	int vd1_x, vd1_y, vd1_z, vd2_x, vd2_y, vd2_z;
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	int ninv_size;
	int lx, ly, lz, rx, ry, rz;
	float max_w1, max_w2;
	FVolume vd1t, vd2t;
	FVolume mask1t, mask2t;
	DVolume vd1gx, vd1gy, vd1gz, vd2gx, vd2gy, vd2gz;
	FVolume vd1w, vd1wx, vd1wy, vd1wz, vd1wx_g, vd1wy_g, vd1wz_g;
	FVolume vd2w, vd2wx, vd2wy, vd2wz, vd2wx_g, vd2wy_g, vd2wz_g;
	int vdt_x, vdt_y, vdt_z;
	//float xc, yc, zc;
	int ixc, iyc, izc;
	int ix, iy, iz;
	float vd1_val, vd2_val;
	float fixedMean, movingMean;
	float suma2, sumb2, suma, sumb, sumab, count;
	double sff, smm, sfm;
	float w1, w1x, w1y, w1z;
	float w2, w2x, w2y, w2z;
	float mask1_val, mask2_val;
	double energy = 0;
#ifdef USE_CC_NCC
	double energy_cc = 0;
#endif
	BOOL bComputeInverse = FALSE;
	//
	double Ii, Ji;
	double localCrossCorrelation;
	double gIx, gIy, gIz;
	double gJx, gJy, gJz;

	vd1_x = vd1.m_vd_x;
	vd1_y = vd1.m_vd_y;
	vd1_z = vd1.m_vd_z;
	vd2_x = vd2.m_vd_x;
	vd2_y = vd2.m_vd_y;
	vd2_z = vd2.m_vd_z;

	if (vd1_x != vd2_x || vd1_y != vd2_y || vd1_z != vd2_z) {
		return 0;
	}
	if (vx_inv != NULL && vy_inv != NULL && vz_inv != NULL) {
		bComputeInverse = TRUE;
	}

	ninv_x = 2 * radius + 1;
	ninv_y = 2 * radius + 1;
	ninv_z = 2 * radius + 1;
	ninv_size = ninv_x * ninv_y * ninv_z;
	//
	ninv_cx = ninv_x / 2;
	ninv_cy = ninv_y / 2;
	ninv_cz = ninv_z / 2;

	lx = ninv_cx;
	rx = ninv_cx+1;
	ly = ninv_cy;
	ry = ninv_cy+1;
	lz = ninv_cz;
	rz = ninv_cz+1;

	vdt_x = max(vd1_x+lx+rx, vd2_x+lx+rx);
	vdt_y = max(vd1_y+ly+ry, vd2_y+ly+ry);
	vdt_z = max(vd1_z+lz+rz, vd2_z+lz+rz);

	vd1t.allocate(vdt_x, vdt_y, vdt_z);
	vd2t.allocate(vdt_x, vdt_y, vdt_z);
	mask1t.allocate(vdt_x, vdt_y, vdt_z);
	mask2t.allocate(vdt_x, vdt_y, vdt_z);
	{
		vd1gx.allocate(vdt_x, vdt_y, vdt_z);
		vd1gy.allocate(vdt_x, vdt_y, vdt_z);
		vd1gz.allocate(vdt_x, vdt_y, vdt_z);
		vd1wx.allocate(vd1_x, vd1_y, vd1_z);
		vd1wy.allocate(vd1_x, vd1_y, vd1_z);
		vd1wz.allocate(vd1_x, vd1_y, vd1_z);
	}
	if (bComputeInverse) {
		vd2gx.allocate(vdt_x, vdt_y, vdt_z);
		vd2gy.allocate(vdt_x, vdt_y, vdt_z);
		vd2gz.allocate(vdt_x, vdt_y, vdt_z);
		vd2wx.allocate(vd2_x, vd2_y, vd2_z);
		vd2wy.allocate(vd2_x, vd2_y, vd2_z);
		vd2wz.allocate(vd2_x, vd2_y, vd2_z);
	}
	
	// translate images
	#pragma omp parallel for private(i,j,k,ixc,iyc,izc)
	for (k = 0; k < vdt_z; k++) {
		for (j = 0; j < vdt_y; j++) {
			for (i = 0; i < vdt_x; i++) {
				ixc = i - lx;
				iyc = j - ly;
				izc = k - lz;
				//if ((ixc <= 0) || (ixc >= vd1_x-1) || (iyc <= 0) || (iyc >= vd1_y-1) || (izc <= 0) || (izc >= vd1_z-1)) {
				if ((ixc < 0) || (ixc > vd1_x-1) || (iyc < 0) || (iyc > vd1_y-1) || (izc < 0) || (izc > vd1_z-1)) {
					vd1t.m_pData[k][j][i][0] = 0;
					vd2t.m_pData[k][j][i][0] = 0;
					//
					mask1t.m_pData[k][j][i][0] = 0;
					mask2t.m_pData[k][j][i][0] = 0;
				} else {
					vd1t.m_pData[k][j][i][0] = vd1.m_pData[izc][iyc][ixc][0];
					vd2t.m_pData[k][j][i][0] = vd2.m_pData[izc][iyc][ixc][0];
					//
					if (dc_skip_back == 1) {
						if (vd1.m_pData[izc][iyc][ixc][0] <= dc_back_color || vd2.m_pData[izc][iyc][ixc][0] <= dc_back_color) {
							mask1t.m_pData[k][j][i][0] = 0;
							mask2t.m_pData[k][j][i][0] = 0;
							continue;
						}
					}
					//
					if (mask1 != NULL) {
						mask1t.m_pData[k][j][i][0] = mask1->m_pData[izc][iyc][ixc][0];
					} else {
						mask1t.m_pData[k][j][i][0] = 1;
					}
					if (mask2 != NULL) {
						mask2t.m_pData[k][j][i][0] = mask2->m_pData[izc][iyc][ixc][0];
					} else {
						mask2t.m_pData[k][j][i][0] = 1;
					}
				}
			}
		}
	}
		
	//vd1t.GetGradient(vd1gx, vd1gy, vd1gz);
	vd1t.GetGradient(vd1gx, vd1gy, vd1gz, mask1t, 0.0f);
	if (bComputeInverse) {
		//vd2t.GetGradient(vd2gx, vd2gy, vd2gz);
		vd2t.GetGradient(vd2gx, vd2gy, vd2gz, mask2t, 0.0f);
	}

#ifdef TEST_GET_VELOCITY_CC
	vd1t.save("vd1t.nii.gz", 1);
	vd2t.save("vd2t.nii.gz", 1);
	mask1t.save("mask1t.nii.gz", 1);
	mask2t.save("mask2t.nii.gz", 1);
	SaveMHDData(NULL, "vd1g.mhd", vd1gx.m_pData, vd1gy.m_pData, vd1gz.m_pData, vdt_x, vdt_y, vdt_z, 1, 1, 1, 0, 0, 0);
	if (bComputeInverse) {
		SaveMHDData(NULL, "vd2g.mhd", vd2gx.m_pData, vd2gy.m_pData, vd2gz.m_pData, vdt_x, vdt_y, vdt_z, 1, 1, 1, 0, 0, 0);
	}
#endif

#ifdef USE_CC_NCC
	#pragma omp parallel for private(l,m,n,ixc,iyc,izc,count,suma,suma2,sumb,sumb2,sumab,i,j,k,ix,iy,iz,vd1_val,vd2_val,fixedMean,movingMean,sff,smm,sfm,Ii,Ji,localCrossCorrelation,gIx,gIy,gIz,gJx,gJy,gJz,w1,w1x,w1y,w1z,w2,w2x,w2y,w2z,mask1_val,mask2_val) shared(energy,energy_cc)
#else
	#pragma omp parallel for private(l,m,n,ixc,iyc,izc,count,suma,suma2,sumb,sumb2,sumab,i,j,k,ix,iy,iz,vd1_val,vd2_val,fixedMean,movingMean,sff,smm,sfm,Ii,Ji,localCrossCorrelation,gIx,gIy,gIz,gJx,gJy,gJz,w1,w1x,w1y,w1z,w2,w2x,w2y,w2z,mask1_val,mask2_val) shared(energy)
#endif
	/*
	for (n = 0; n < vd1_z; n++) {
		for (m = 0; m < vd1_y; m++) {
			for (l = 0; l < vd1_x; l++) {
	/*/
	for (n = radius; n < vd1_z-radius; n++) {
		for (m = radius; m < vd1_y-radius; m++) {
			for (l = radius; l < vd1_x-radius; l++) {
	//*/
				ixc = (int)(l - ninv_cx + lx);
				iyc = (int)(m - ninv_cy + ly);
				izc = (int)(n - ninv_cz + lz);
				//
				w2x = w2y = w2z = 0; // to prevent compiler warning
				//
				count = 0;
				suma2 = suma = 0;
				sumb2 = sumb = 0;
				sumab = 0;
				for (k = 0; k < ninv_z; k++) {
					for (j = 0; j < ninv_y; j++) {
						for (i = 0; i < ninv_x; i++) {
							ix = ixc + i;
							iy = iyc + j;
							iz = izc + k;
							//
							if (mask1t.m_pData[iz][iy][ix][0] > 0 && mask2t.m_pData[iz][iy][ix][0] > 0) {
								vd1_val = vd1t.m_pData[iz][iy][ix][0];
								vd2_val = vd2t.m_pData[iz][iy][ix][0];
								//
								suma  += vd1_val;
								suma2 += vd1_val * vd1_val;
								sumb  += vd2_val;
								sumb2 += vd2_val * vd2_val;
								sumab += vd1_val * vd2_val;
								count += 1;
							}
						}
					}
				}
				//
				ixc = l + lx;
				iyc = m + ly;
				izc = n + lz;
				//
				if (count > 0) {
					fixedMean  = suma / count;
					movingMean = sumb / count;
					sff = suma2 -  fixedMean*suma -  fixedMean*suma + count* fixedMean* fixedMean;
					smm = sumb2 - movingMean*sumb - movingMean*sumb + count*movingMean*movingMean;
					sfm = sumab - movingMean*suma -  fixedMean*sumb + count*movingMean* fixedMean;
					//
					if ((sff > 0) && (smm > 0)) {
					//if ((sff > 1e-1) && (smm > 1e-1)) {
					//if (sff*smm > 1.e-5) {

						if (sff*smm > 1.e-5) {
#ifdef USE_CC_NCC
							localCrossCorrelation = sfm / sqrt(sff * smm);
							if (localCrossCorrelation <= 1.0) {
								#pragma omp atomic
								energy += 1.0 - localCrossCorrelation;
							}
							//
							localCrossCorrelation = sfm*sfm / (sff * smm);
							if (localCrossCorrelation <= 1.0) {
								#pragma omp atomic
								energy_cc -= localCrossCorrelation;
							}
#else
							localCrossCorrelation = sfm*sfm / (sff * smm);
							if (localCrossCorrelation <= 1.0) {
								#pragma omp atomic
								energy -= localCrossCorrelation;
							}
#endif

						}

						Ii = vd1t.m_pData[izc][iyc][ixc][0] - fixedMean;
						Ji = vd2t.m_pData[izc][iyc][ixc][0] - movingMean;

						{
							gIx = vd1gx.m_pData[izc][iyc][ixc][0];
							gIy = vd1gy.m_pData[izc][iyc][ixc][0];
							gIz = vd1gz.m_pData[izc][iyc][ixc][0];
#ifdef USE_CC_NCC
							w1 = - 1.0 / sqrt(sff*smm) * ( Ji - sfm/sff * Ii);
#else
							w1 = - 2.0 * sfm / (sff*smm) * (Ji - sfm/sff * Ii);
#endif
							w1x = w1 * gIx;
							w1y = w1 * gIy;
							w1z = w1 * gIz;
						}
						if (bComputeInverse) {
							gJx = vd2gx.m_pData[izc][iyc][ixc][0];
							gJy = vd2gy.m_pData[izc][iyc][ixc][0];
							gJz = vd2gz.m_pData[izc][iyc][ixc][0];
#ifdef USE_CC_NCC
							w2 = - 1.0 / sqrt(sff*smm) * ( Ii - sfm/smm * Ji);
#else
							w2 = - 2.0 * sfm / (sff*smm) * (Ii - sfm/smm * Ji);
#endif
							w2x = w2 * gJx;
							w2y = w2 * gJy;
							w2z = w2 * gJz;
						}
					} else {
						w1x = w1y = w1z = 0;
						if (bComputeInverse) {
							w2x = w2y = w2z = 0;
						}
					}
				} else {
					w1x = w1y = w1z = 0;
					if (bComputeInverse) {
						w2x = w2y = w2z = 0;
					}
				}
				//
				mask1_val = mask1t.m_pData[izc][iyc][ixc][0];
				//
				vd1wx.m_pData[n][m][l][0] = w1x * mask1_val;
				vd1wy.m_pData[n][m][l][0] = w1y * mask1_val;
				vd1wz.m_pData[n][m][l][0] = w1z * mask1_val;
				//
				if (bComputeInverse) {
					mask2_val = mask2t.m_pData[izc][iyc][ixc][0];
					//
					vd2wx.m_pData[n][m][l][0] = w2x * mask2_val;
					vd2wy.m_pData[n][m][l][0] = w2y * mask2_val;
					vd2wz.m_pData[n][m][l][0] = w2z * mask2_val;
				}
			}
		}
	}

#ifdef USE_CC_NCC
	TRACE2("energy_cc = %f\n", energy_cc);
#endif

	vd1t.clear();
	vd2t.clear();
	mask1t.clear();
	mask2t.clear();
	{
		vd1gx.clear();
		vd1gy.clear();
		vd1gz.clear();
	}
	if (bComputeInverse) {
		vd2gx.clear();
		vd2gy.clear();
		vd2gz.clear();
	}

#ifdef TEST_GET_VELOCITY_CC
	SaveMHDData(NULL, "vd1w.mhd", vd1wx.m_pData, vd1wy.m_pData, vd1wz.m_pData, vd1_x, vd1_y, vd1_z, 1, 1, 1, 0, 0, 0);
	if (bComputeInverse) {
		SaveMHDData(NULL, "vd2w.mhd", vd2wx.m_pData, vd2wy.m_pData, vd2wz.m_pData, vd2_x, vd2_y, vd2_z, 1, 1, 1, 0, 0, 0);
	}
#endif

	{
		vd1wx_g.allocate(vd1_x, vd1_y, vd1_z);
		vd1wy_g.allocate(vd1_x, vd1_y, vd1_z);
		vd1wz_g.allocate(vd1_x, vd1_y, vd1_z);

		/*
		vd1wx.GaussianSmoothing(vd1wx_g, sigma, 5);
		vd1wy.GaussianSmoothing(vd1wy_g, sigma, 5);
		vd1wz.GaussianSmoothing(vd1wz_g, sigma, 5);
		/*/
		SmoothField(vd1wx, vd1wy, vd1wz, vd1wx_g, vd1wy_g, vd1wz_g, sigma);
		//*/

		vd1wx.clear();
		vd1wy.clear();
		vd1wz.clear();
	}
	if (bComputeInverse) {
		vd2wx_g.allocate(vd2_x, vd2_y, vd2_z);
		vd2wy_g.allocate(vd2_x, vd2_y, vd2_z);
		vd2wz_g.allocate(vd2_x, vd2_y, vd2_z);

		/*
		vd2wx.GaussianSmoothing(vd2wx_g, sigma, 5);
		vd2wy.GaussianSmoothing(vd2wy_g, sigma, 5);
		vd2wz.GaussianSmoothing(vd2wz_g, sigma, 5);
		/*/
		SmoothField(vd2wx, vd2wy, vd2wz, vd2wx_g, vd2wy_g, vd2wz_g, sigma);
		//*/

		vd2wx.clear();
		vd2wy.clear();
		vd2wz.clear();
	}

#ifdef TEST_GET_VELOCITY_CC
	SaveMHDData(NULL, "vd1w_g.mhd", vd1wx_g.m_pData, vd1wy_g.m_pData, vd1wz_g.m_pData, vd1_x, vd1_y, vd1_z, 1, 1, 1, 0, 0, 0);
	if (bComputeInverse) {
		SaveMHDData(NULL, "vd2w_g.mhd", vd2wx_g.m_pData, vd2wy_g.m_pData, vd2wz_g.m_pData, vd2_x, vd2_y, vd2_z, 1, 1, 1, 0, 0, 0);
	}
#endif

	max_w1 = 0;
	max_w2 = 0;
	#pragma omp parallel for private(l,m,n,w1,w1x,w1y,w1z,w2,w2x,w2y,w2z) shared(max_w1,max_w2)
	for (n = 0; n < vd1_z; n++) {
		for (m = 0; m < vd1_y; m++) {
			for (l = 0; l < vd1_x; l++) {
				{
					w1x = vd1wx_g.m_pData[n][m][l][0];
					w1y = vd1wy_g.m_pData[n][m][l][0];
					w1z = vd1wz_g.m_pData[n][m][l][0];
					w1 = sqrt(w1x*w1x + w1y*w1y + w1z*w1z);
					#pragma omp critical
					{
						if (w1 > max_w1) { max_w1 = w1; }
					}
				}
				if (bComputeInverse) {
					w2x = vd2wx_g.m_pData[n][m][l][0];
					w2y = vd2wy_g.m_pData[n][m][l][0];
					w2z = vd2wz_g.m_pData[n][m][l][0];
					w2 = sqrt(w2x*w2x + w2y*w2y + w2z*w2z);
					#pragma omp critical
					{
						if (w2 > max_w2) { max_w2 = w2; }
					}
				}
			}
		}
	}
	/*
	TRACE2("max_w1 = %f\n", max_w1);
	if (bComputeInverse) {
		TRACE2("max_w2 = %f\n", max_w2);
	}
	//*/
	if (max_w1 <= 0) { max_w1 = 1; }
	if (max_w2 <= 0) { max_w2 = 1; }

	#pragma omp parallel for private(l,m,n)
	for (n = 0; n < vd1_z; n++) {
		for (m = 0; m < vd1_y; m++) {
			for (l = 0; l < vd1_x; l++) {
				{
					vx->m_pData[n][m][l][0] = weight * vd1wx_g.m_pData[n][m][l][0] / max_w1;
					vy->m_pData[n][m][l][0] = weight * vd1wy_g.m_pData[n][m][l][0] / max_w1;
					vz->m_pData[n][m][l][0] = weight * vd1wz_g.m_pData[n][m][l][0] / max_w1;
				}
				if (bComputeInverse) {
					vx_inv->m_pData[n][m][l][0] = weight * vd2wx_g.m_pData[n][m][l][0] / max_w2;
					vy_inv->m_pData[n][m][l][0] = weight * vd2wy_g.m_pData[n][m][l][0] / max_w2;
					vz_inv->m_pData[n][m][l][0] = weight * vd2wz_g.m_pData[n][m][l][0] / max_w2;
				}
			}
		}
	}

#if 0
	max_w1 = 0;
	max_w2 = 0;
	for (n = 0; n < vd1_z; n++) {
		for (m = 0; m < vd1_y; m++) {
			for (l = 0; l < vd1_x; l++) {
				{
					w1x = vx->m_pData[n][m][l][0];
					w1y = vy->m_pData[n][m][l][0];
					w1z = vz->m_pData[n][m][l][0];
					w1 = sqrt(w1x*w1x + w1y*w1y + w1z*w1z);
					if (w1 > max_w1) { max_w1 = w1; }
				}
				if (bComputeInverse) {
					w2x = vx_inv->m_pData[n][m][l][0];
					w2y = vy_inv->m_pData[n][m][l][0];
					w2z = vz_inv->m_pData[n][m][l][0];
					w2 = sqrt(w2x*w2x + w2y*w2y + w2z*w2z);
					if (w2 > max_w2) { max_w2 = w2; }
				}
			}
		}
	}
	/*
	TRACE2("max_w1 = %f\n", max_w1);
	if (bComputeInverse) {
		TRACE2("max_w2 = %f\n", max_w2);
	}
	//*/
#endif

#ifdef TEST_GET_VELOCITY_CC
	SaveMHDData(NULL, "vd1w_g_n.mhd", vx->m_pData, vy->m_pData, vz->m_pData, vd1_x, vd1_y, vd1_z, 1, 1, 1, 0, 0, 0);
	if (bComputeInverse) {
		SaveMHDData(NULL, "vd2w_g_n.mhd", vx_inv->m_pData, vy_inv->m_pData, vz_inv->m_pData, vd2_x, vd2_y, vd2_z, 1, 1, 1, 0, 0, 0);
	}
#endif

	{
		vd1wx_g.clear();
		vd1wy_g.clear();
		vd1wz_g.clear();
	}
	if (bComputeInverse) {
		vd2wx_g.clear();
		vd2wy_g.clear();
		vd2wz_g.clear();
	}

	return energy;
}

#if !defined(USE_TE_L2) && !defined(USE_TE_JS)
#define USE_TE_L2
#endif
//#define TEST_GET_VELOCITY_TUMOR
void GetVelocity_Tumor(FVolume& tu1, FVolume& tu2, FVolume* vx, FVolume* vy, FVolume* vz, FVolume* vx_inv, FVolume* vy_inv, FVolume* vz_inv, float weight = 0.2f, float sigma = 3.0f) 
{
	int l, m, n;
	int vd1_x, vd1_y, vd1_z, vd2_x, vd2_y, vd2_z;
	float max_w1, max_w2;
	DVolume vd1gx, vd1gy, vd1gz, vd2gx, vd2gy, vd2gz;
	FVolume vd1w, vd1wx, vd1wy, vd1wz, vd1wx_g, vd1wy_g, vd1wz_g;
	FVolume vd2w, vd2wx, vd2wy, vd2wz, vd2wx_g, vd2wy_g, vd2wz_g;
	int ixc, iyc, izc;
	float w1, w1x, w1y, w1z;
	float w2, w2x, w2y, w2z;
	BOOL bComputeInverse = FALSE;
	//
	double Ii, Ji;
#ifdef USE_TE_JS
	double Mi;
#endif
	double gIx, gIy, gIz;
	double gJx, gJy, gJz;

	vd1_x = tu1.m_vd_x;
	vd1_y = tu1.m_vd_y;
	vd1_z = tu1.m_vd_z;
	vd2_x = tu2.m_vd_x;
	vd2_y = tu2.m_vd_y;
	vd2_z = tu2.m_vd_z;

	if (vd1_x != vd2_x || vd1_y != vd2_y || vd1_z != vd2_z) {
		return;
	}
	if (vx_inv != NULL && vy_inv != NULL && vz_inv != NULL) {
		bComputeInverse = TRUE;
	}

	{
		vd1gx.allocate(vd1_x, vd1_y, vd1_z);
		vd1gy.allocate(vd1_x, vd1_y, vd1_z);
		vd1gz.allocate(vd1_x, vd1_y, vd1_z);
		vd1wx.allocate(vd1_x, vd1_y, vd1_z);
		vd1wy.allocate(vd1_x, vd1_y, vd1_z);
		vd1wz.allocate(vd1_x, vd1_y, vd1_z);
	}
	if (bComputeInverse) {
		vd2gx.allocate(vd2_x, vd2_y, vd2_z);
		vd2gy.allocate(vd2_x, vd2_y, vd2_z);
		vd2gz.allocate(vd2_x, vd2_y, vd2_z);
		vd2wx.allocate(vd2_x, vd2_y, vd2_z);
		vd2wy.allocate(vd2_x, vd2_y, vd2_z);
		vd2wz.allocate(vd2_x, vd2_y, vd2_z);
	}
			
	tu1.GetGradient(vd1gx, vd1gy, vd1gz);
	if (bComputeInverse) {
		tu2.GetGradient(vd2gx, vd2gy, vd2gz);
	}

#ifdef TEST_GET_VELOCITY_TUMOR
	SaveMHDData(NULL, "vd1g.mhd", vd1gx.m_pData, vd1gy.m_pData, vd1gz.m_pData, vd1_x, vd1_y, vd1_z, 1, 1, 1, 0, 0, 0);
	if (bComputeInverse) {
		SaveMHDData(NULL, "vd2g.mhd", vd2gx.m_pData, vd2gy.m_pData, vd2gz.m_pData, vd2_x, vd2_y, vd2_z, 1, 1, 1, 0, 0, 0);
	}
#endif

#ifdef USE_TE_JS
	#pragma omp parallel for private(l,m,n,ixc,iyc,izc,Ii,Ji,Mi,gIx,gIy,gIz,w1,gJx,gJy,gJz,w2)
#else
	#pragma omp parallel for private(l,m,n,ixc,iyc,izc,Ii,Ji,gIx,gIy,gIz,w1,gJx,gJy,gJz,w2)
#endif
	for (n = 0; n < vd1_z; n++) {
		for (m = 0; m < vd1_y; m++) {
			for (l = 0; l < vd1_x; l++) {
				ixc = l;
				iyc = m;
				izc = n;
				//
				Ii = tu1.m_pData[izc][iyc][ixc][0];
				Ji = tu2.m_pData[izc][iyc][ixc][0];
#ifdef USE_TE_JS
				Mi = 0.5 * (Ii + Ji);
#endif
				//
				{
					gIx = vd1gx.m_pData[izc][iyc][ixc][0];
					gIy = vd1gy.m_pData[izc][iyc][ixc][0];
					gIz = vd1gz.m_pData[izc][iyc][ixc][0];

#ifdef USE_TE_L2
					w1 = 2.0 * (Ii -  Ji);
#endif
#ifdef USE_TE_JS
					w1 = 0.5 * (ENT2D(Mi) - ENT2D(Ii));
#endif

					vd1wx.m_pData[n][m][l][0] = w1 * gIx;
					vd1wy.m_pData[n][m][l][0] = w1 * gIy;
					vd1wz.m_pData[n][m][l][0] = w1 * gIz;
				}
				if (bComputeInverse) {
					gJx = vd2gx.m_pData[izc][iyc][ixc][0];
					gJy = vd2gy.m_pData[izc][iyc][ixc][0];
					gJz = vd2gz.m_pData[izc][iyc][ixc][0];

#ifdef USE_TE_L2
					w2 = 2.0 * (Ji - Ii);
#endif
#ifdef USE_TE_JS
					w2 = 0.5 * (ENT2D(Mi) - ENT2D(Ji));
#endif

					vd2wx.m_pData[n][m][l][0] = w2 * gJx;
					vd2wy.m_pData[n][m][l][0] = w2 * gJy;
					vd2wz.m_pData[n][m][l][0] = w2 * gJz;
				}
			}
		}
	}

	{
		vd1gx.clear();
		vd1gy.clear();
		vd1gz.clear();
	}
	if (bComputeInverse) {
		vd2gx.clear();
		vd2gy.clear();
		vd2gz.clear();
	}

#ifdef TEST_GET_VELOCITY_TUMOR
	SaveMHDData(NULL, "vd1w.mhd", vd1wx.m_pData, vd1wy.m_pData, vd1wz.m_pData, vd1_x, vd1_y, vd1_z, 1, 1, 1, 0, 0, 0);
	if (bComputeInverse) {
		SaveMHDData(NULL, "vd2w.mhd", vd2wx.m_pData, vd2wy.m_pData, vd2wz.m_pData, vd2_x, vd2_y, vd2_z, 1, 1, 1, 0, 0, 0);
	}
#endif

	{
		vd1wx_g.allocate(vd1_x, vd1_y, vd1_z);
		vd1wy_g.allocate(vd1_x, vd1_y, vd1_z);
		vd1wz_g.allocate(vd1_x, vd1_y, vd1_z);

		/*
		vd1wx.GaussianSmoothing(vd1wx_g, sigma, 5);
		vd1wy.GaussianSmoothing(vd1wy_g, sigma, 5);
		vd1wz.GaussianSmoothing(vd1wz_g, sigma, 5);
		/*/
		SmoothField(vd1wx, vd1wy, vd1wz, vd1wx_g, vd1wy_g, vd1wz_g, sigma);
		//*/

		vd1wx.clear();
		vd1wy.clear();
		vd1wz.clear();
	}
	if (bComputeInverse) {
		vd2wx_g.allocate(vd2_x, vd2_y, vd2_z);
		vd2wy_g.allocate(vd2_x, vd2_y, vd2_z);
		vd2wz_g.allocate(vd2_x, vd2_y, vd2_z);

		/*
		vd2wx.GaussianSmoothing(vd2wx_g, sigma, 5);
		vd2wy.GaussianSmoothing(vd2wy_g, sigma, 5);
		vd2wz.GaussianSmoothing(vd2wz_g, sigma, 5);
		/*/
		SmoothField(vd2wx, vd2wy, vd2wz, vd2wx_g, vd2wy_g, vd2wz_g, sigma);
		//*/

		vd2wx.clear();
		vd2wy.clear();
		vd2wz.clear();
	}

#ifdef TEST_GET_VELOCITY_TUMOR
	SaveMHDData(NULL, "vd1w_g.mhd", vd1wx_g.m_pData, vd1wy_g.m_pData, vd1wz_g.m_pData, vd1_x, vd1_y, vd1_z, 1, 1, 1, 0, 0, 0);
	if (bComputeInverse) {
		SaveMHDData(NULL, "vd2w_g.mhd", vd2wx_g.m_pData, vd2wy_g.m_pData, vd2wz_g.m_pData, vd2_x, vd2_y, vd2_z, 1, 1, 1, 0, 0, 0);
	}
#endif

	max_w1 = 0;
	max_w2 = 0;
	#pragma omp parallel for private(l,m,n,w1x,w1y,w1z,w1,w2x,w2y,w2z,w2) shared(max_w1,max_w2)
	for (n = 0; n < vd1_z; n++) {
		for (m = 0; m < vd1_y; m++) {
			for (l = 0; l < vd1_x; l++) {
				{
					w1x = vd1wx_g.m_pData[n][m][l][0];
					w1y = vd1wy_g.m_pData[n][m][l][0];
					w1z = vd1wz_g.m_pData[n][m][l][0];
					w1 = sqrt(w1x*w1x + w1y*w1y + w1z*w1z);
					#pragma omp critical
					{
						if (w1 > max_w1) { max_w1 = w1; }
					}
				}
				if (bComputeInverse) {
					w2x = vd2wx_g.m_pData[n][m][l][0];
					w2y = vd2wy_g.m_pData[n][m][l][0];
					w2z = vd2wz_g.m_pData[n][m][l][0];
					w2 = sqrt(w2x*w2x + w2y*w2y + w2z*w2z);
					#pragma omp critical
					{
						if (w2 > max_w2) { max_w2 = w2; }
					}
				}
			}
		}
	}
	/*
	TRACE2("max_w1 = %f\n", max_w1);
	if (bComputeInverse) {
		TRACE2("max_w2 = %f\n", max_w2);
	}
	//*/
	if (max_w1 <= 0) { max_w1 = 1; }
	if (max_w2 <= 0) { max_w2 = 1; }

	#pragma omp parallel for private(l,m,n)
	for (n = 0; n < vd1_z; n++) {
		for (m = 0; m < vd1_y; m++) {
			for (l = 0; l < vd1_x; l++) {
				{
					vx->m_pData[n][m][l][0] = weight * vd1wx_g.m_pData[n][m][l][0] / max_w1;
					vy->m_pData[n][m][l][0] = weight * vd1wy_g.m_pData[n][m][l][0] / max_w1;
					vz->m_pData[n][m][l][0] = weight * vd1wz_g.m_pData[n][m][l][0] / max_w1;
				}
				if (bComputeInverse) {
					vx_inv->m_pData[n][m][l][0] = weight * vd2wx_g.m_pData[n][m][l][0] / max_w2;
					vy_inv->m_pData[n][m][l][0] = weight * vd2wy_g.m_pData[n][m][l][0] / max_w2;
					vz_inv->m_pData[n][m][l][0] = weight * vd2wz_g.m_pData[n][m][l][0] / max_w2;
				}
			}
		}
	}

#if 0
	max_w1 = 0;
	max_w2 = 0;
	for (n = 0; n < vd1_z; n++) {
		for (m = 0; m < vd1_y; m++) {
			for (l = 0; l < vd1_x; l++) {
				{
					w1x = vx->m_pData[n][m][l][0];
					w1y = vy->m_pData[n][m][l][0];
					w1z = vz->m_pData[n][m][l][0];
					w1 = sqrt(w1x*w1x + w1y*w1y + w1z*w1z);
					if (w1 > max_w1) { max_w1 = w1; }
				}
				if (bComputeInverse) {
					w2x = vx_inv->m_pData[n][m][l][0];
					w2y = vy_inv->m_pData[n][m][l][0];
					w2z = vz_inv->m_pData[n][m][l][0];
					w2 = sqrt(w2x*w2x + w2y*w2y + w2z*w2z);
					if (w2 > max_w2) { max_w2 = w2; }
				}
			}
		}
	}
	/*
	TRACE2("max_w1 = %f\n", max_w1);
	if (bComputeInverse) {
		TRACE2("max_w2 = %f\n", max_w2);
	}
	//*/
#endif

#ifdef TEST_GET_VELOCITY_TUMOR
	SaveMHDData(NULL, "vd1w_g_n.mhd", vx->m_pData, vy->m_pData, vz->m_pData, vd1_x, vd1_y, vd1_z, 1, 1, 1, 0, 0, 0);
	if (bComputeInverse) {
		SaveMHDData(NULL, "vd2w_g_n.mhd", vx_inv->m_pData, vy_inv->m_pData, vz_inv->m_pData, vd2_x, vd2_y, vd2_z, 1, 1, 1, 0, 0, 0);
	}
#endif

	{
		vd1wx_g.clear();
		vd1wy_g.clear();
		vd1wz_g.clear();
	}
	if (bComputeInverse) {
		vd2wx_g.clear();
		vd2wy_g.clear();
		vd2wz_g.clear();
	}
}

BOOL CheckConvergence(double* energy_list, int num_list)
{
	typedef itk::Vector<double, 1> ProfilePointDataType;
	typedef itk::Image<ProfilePointDataType, 1> CurveType;
	typedef itk::PointSet<ProfilePointDataType, 1> EnergyProfileType;
	typedef EnergyProfileType::PointType ProfilePointType;
	typedef itk::BSplineScatteredDataPointSetToImageFilter <EnergyProfileType, CurveType> BSplinerType;
	ProfilePointType point;
	ProfilePointDataType energy;
	BSplinerType::Pointer bspliner;
	CurveType::PointType origin;  
	CurveType::SizeType size; 
	CurveType::SpacingType spacing; 
	BSplinerType::ArrayType ncps;
	EnergyProfileType::Pointer window;
	ProfilePointType endPoint;
	int dstart, dsize, wstart;
	double Et, Eg;
	int i;

	dsize = 12;
	if (num_list <= dsize) {
		return FALSE;
	}

	EnergyProfileType::Pointer energyProfile = EnergyProfileType::New();
	energyProfile->Initialize();
	for (i = 0; i < num_list; i++) {
		point[0] = i;
		energy[0] = energy_list[i];
		energyProfile->SetPoint(i, point);
		energyProfile->SetPointData(i, energy);
	}

	bspliner = BSplinerType::New();

	dstart = num_list - dsize;
	origin.Fill(dstart);
	size.Fill(dsize);
	spacing.Fill(1);

	window = EnergyProfileType::New();
	window->Initialize();     

	wstart = (int)origin[0];
	Et = 0;
	for (i = wstart; i < num_list; i++) {
		point[0] = i;
		energy.Fill(0);
		energyProfile->GetPointData(i, &energy);
		Et += energy[0];
		window->SetPoint(i - wstart, point);
		window->SetPointData(i - wstart, energy);
	}   
	if (Et > 0) Et *= (-1.0);
	for (i = wstart; i < num_list; i++) {
		energy.Fill(0);
		energyProfile->GetPointData(i, &energy);
		window->SetPointData(i-wstart, energy / Et);
	}

	bspliner->SetInput(window);
	bspliner->SetOrigin(origin);
	bspliner->SetSpacing(spacing);
	bspliner->SetSize(size);
	bspliner->SetNumberOfLevels(1);
	bspliner->SetSplineOrder(1); 
	// single span, order = 2
	ncps.Fill(2);
	bspliner->SetNumberOfControlPoints(ncps);  
	bspliner->Update();

#if ITK_VERSION_MAJOR >= 4
	typedef itk::BSplineControlPointImageFunction<CurveType> BSplinerFunctionType;
	BSplinerFunctionType::Pointer bsplinerFunction = BSplinerFunctionType::New();
	bsplinerFunction->SetOrigin(origin);
	bsplinerFunction->SetSpacing(spacing);
	bsplinerFunction->SetSize(size);
	bsplinerFunction->SetSplineOrder(bspliner->GetSplineOrder());
	bsplinerFunction->SetInputImage(bspliner->GetPhiLattice());
	endPoint[0] = (double)(num_list - dsize * 0.5);
	BSplinerFunctionType::GradientType gradient = bsplinerFunction->EvaluateGradientAtParametricPoint(endPoint);
#else
	BSplinerType::GradientType gradient;
	endPoint[0] = (double)(num_list - dsize * 0.5);
	gradient.Fill(0);
	bspliner->EvaluateGradientAtPoint(endPoint, gradient); 
#endif
	Eg = gradient[0][0];

	TRACE2("Eg = %e\n", Eg);

	if (Eg < 0.0001) {
		return TRUE;
	} else {
		return FALSE;
	}
}

ImagePointer WarpImage(ImagePointer referenceimage, ImagePointer movingImage, DeformationFieldPointer totalField, float p_val = 0)
{
	typedef itk::WarpImageFilter<ImageType, ImageType, DeformationFieldType> WarperType;
	typedef WarperType::CoordRepType CoordRepType;
	typedef itk::LinearInterpolateImageFunction<ImageType, CoordRepType> InterpolatorType;

    WarperType::Pointer warper = WarperType::New();
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    warper->SetInterpolator(interpolator);
    warper->SetInput(movingImage);
#if ITK_VERSION_MAJOR >= 4
    warper->SetDisplacementField(totalField);
#else
    warper->SetDeformationField(totalField);
#endif
    warper->SetOutputSpacing(referenceimage->GetSpacing());
    warper->SetOutputOrigin(referenceimage->GetOrigin());
    warper->SetOutputDirection(referenceimage->GetDirection());

    try {
        warper->Update();
    } catch (itk::ExceptionObject& e) {
        std::cerr << e << std::endl;
        exit(EXIT_FAILURE);
    }

	ImageType::Pointer outimg = warper->GetOutput();

	return outimg;
}

BOOL WarpImage(FVolume& img, FVolume& img_out, FVolume& vx, FVolume& vy, FVolume& vz, float p_val = 0)
{
	ImageType::SizeType size;
	ImageType::IndexType start;
	ImageType::RegionType region;
	ImageType::SpacingType spacing;
	double origin[ImageDimension];
    VectorType zero;
    zero.Fill(0);

	img_out.copy(img);

	size[0] = vx.m_vd_x;
	size[1] = vx.m_vd_y;
	size[2] = vx.m_vd_z;
	start[0] = 0;
	start[1] = 0;
	start[2] = 0;
	region.SetSize(size);
	region.SetIndex(start);
	//
	spacing[0] = vx.m_vd_dx;
	spacing[1] = vx.m_vd_dy;
	spacing[2] = vx.m_vd_dz;
	origin[0] = 0.0;
	origin[1] = 0.0;
	origin[2] = 0.0;
	
	ImagePointer refImage;
	ImagePointer movImage;
	ImagePointer outImage;
	DeformationFieldPointer field;

    refImage = ImageType::New();
    refImage->SetRegions(region);
    refImage->SetSpacing(spacing);
	refImage->SetOrigin(origin);
    refImage->Allocate();

    movImage = ImageType::New();
    movImage->SetRegions(region);
    movImage->SetSpacing(spacing);
	movImage->SetOrigin(origin);
    movImage->Allocate();

    field = DeformationFieldType::New();
    field->SetRegions(region);
    field->SetSpacing(spacing);
	field->SetOrigin(origin);
    field->Allocate();
    field->FillBuffer(zero);

	itk::ImageRegionIteratorWithIndex<ImageType> r_iter(refImage, refImage->GetLargestPossibleRegion());
	for (r_iter.GoToBegin(); !r_iter.IsAtEnd(); ++r_iter)
	{
		ImageType::IndexType index = r_iter.GetIndex();
		r_iter.Set(img.m_pData[index[2]][index[1]][index[0]][0]);
	}

	itk::ImageRegionIteratorWithIndex<ImageType> m_iter(movImage, movImage->GetLargestPossibleRegion());
	for (m_iter.GoToBegin(); !m_iter.IsAtEnd(); ++m_iter)
	{
		ImageType::IndexType index = m_iter.GetIndex();
		m_iter.Set(img.m_pData[index[2]][index[1]][index[0]][0]);
	}

	itk::ImageRegionIteratorWithIndex<DeformationFieldType> f_iter(field, field->GetLargestPossibleRegion());
	for (f_iter.GoToBegin(); !f_iter.IsAtEnd(); ++f_iter)
	{
		DeformationFieldType::IndexType index = f_iter.GetIndex();
		VectorType vec;

		vec[0] = vx.m_pData[index[2]][index[1]][index[0]][0];
		vec[1] = vy.m_pData[index[2]][index[1]][index[0]][0];
		vec[2] = vz.m_pData[index[2]][index[1]][index[0]][0];

		f_iter.Set(vec);
	}

	outImage = WarpImage(refImage, movImage, field, p_val);

	itk::ImageRegionIteratorWithIndex<ImageType> i_iter(outImage, outImage->GetLargestPossibleRegion());
	for (i_iter.GoToBegin(); !i_iter.IsAtEnd(); ++i_iter)
	{
		ImageType::IndexType index = i_iter.GetIndex();
		img_out.m_pData[index[2]][index[1]][index[0]][0] = i_iter.Get();
	}

	return TRUE;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
BOOL ComputeDataCost3D_PP_NCC(FVolume* vd1, FVolume* vd2, int d_num, FVolume& mask1, FVolume& mask2,
	REALV**** X1, REALV**** Y1, REALV**** Z1, REALV**** dX1, REALV**** dY1, REALV**** dZ1, REALV**** X2, REALV**** Y2, REALV**** Z2, REALV**** dX2, REALV**** dY2, REALV**** dZ2,
	int mesh_x, int mesh_y, int mesh_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV* disp_x, REALV* disp_y, REALV* disp_z, int num_d, REALV**** dcv,
	int dc_weight, int dc_skip_back, float dc_back_color, int ninv_s = 1)
{
	BOOL res = TRUE;
	//
	float*** ninv;
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	float ninv_size;
	//
	int vd_x, vd_y, vd_z;
	float vd_dx, vd_dy, vd_dz;
	//
	float*** vd1_patch[NumberOfImageChannels];
	float*** vd2_patch[NumberOfImageChannels];
	float*** vd1_patch_mask;
	float*** vd2_patch_mask;
	const float INVALID_VALUE = -FLT_MAX;
	//
	int i, j, k, l, m, n, d, c;
	//
	double HistMin, HistMax;
	double HistInterval;
	double* pHistogramBuffer;
	int nBins = 20000;
	int total = 0; // total is the total number of plausible matches, used to normalize the histogram
	double Prob;
	double MatchingScoreArr[10];
	double ProbArr[10] = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
	int ii, jj;

	// assume 4 channels for image
	if (d_num > NumberOfImageChannels) {
		return FALSE;
	}

	TRACE2("ComputeDataCost3D_PP_NCC...\n");

	vd_x = vd1[0].m_vd_x;
	vd_y = vd1[0].m_vd_y;
	vd_z = vd1[0].m_vd_z;
	vd_dx = vd1[0].m_vd_dx;
	vd_dy = vd1[0].m_vd_dy;
	vd_dz = vd1[0].m_vd_dz;

	{
		ninv_x = 4;
		ninv_y = 4;
		ninv_z = 4;
		ninv_cx = ninv_x / 2;
		ninv_cy = ninv_y / 2;
		ninv_cz = ninv_z / 2;
		//
		ninv_size = ninv_x * ninv_y * ninv_z;
		ninv = (float***)malloc(ninv_z * sizeof(float**));
		for (k = 0; k < ninv_z; k++) {
			ninv[k] = (float**)malloc(ninv_y * sizeof(float*));
			for (j = 0; j < ninv_y; j++) {
				ninv[k][j] = (float*)malloc(ninv_x * sizeof(float));
			}
		}
		get_ninv(ninv, ninv_x / 4, ninv_y / 4, ninv_z / 4);
	}

	for (c = 0; c < d_num; c++) {
		vd1_patch[c] = (float***)malloc(ninv_z * sizeof(float**));
		vd2_patch[c] = (float***)malloc(ninv_z * sizeof(float**));
		for (k = 0; k < ninv_z; k++) {
			vd1_patch[c][k] = (float**)malloc(ninv_y * sizeof(float*));
			vd2_patch[c][k] = (float**)malloc(ninv_y * sizeof(float*));
			for (j = 0; j < ninv_y; j++) {
				vd1_patch[c][k][j] = (float*)malloc(ninv_x * sizeof(float));
				vd2_patch[c][k][j] = (float*)malloc(ninv_x * sizeof(float));
			}
		}
	}
	vd1_patch_mask = (float***)malloc(ninv_z * sizeof(float**));
	vd2_patch_mask = (float***)malloc(ninv_z * sizeof(float**));
	for (k = 0; k < ninv_z; k++) {
		vd1_patch_mask[k] = (float**)malloc(ninv_y * sizeof(float*));
		vd2_patch_mask[k] = (float**)malloc(ninv_y * sizeof(float*));
		for (j = 0; j < ninv_y; j++) {
			vd1_patch_mask[k][j] = (float*)malloc(ninv_x * sizeof(float));
			vd2_patch_mask[k][j] = (float*)malloc(ninv_x * sizeof(float));
		}
	}

	pHistogramBuffer = new double[nBins];
	memset(pHistogramBuffer, 0, sizeof(double) * nBins);
	HistMin = 10e10;
	HistMax = 0;

	for (n = 0; n < mesh_z; n++) {
		TRACE2("processing %d / %d total z\n", n, mesh_z);
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				float x20, y20, z20;
				float x2, y2, z2;
				float x10, y10, z10;
				float x1, y1, z1;
				float dx, dy, dz;
				float ys;
				int ix1, iy1, iz1;
				int ix2, iy2, iz2;
				float fx, fy, fz, fx1, fy1, fz1;
				//
				float ninv_val;
				float vd1_val, vd2_val;
				double vd1_m_c[NumberOfImageChannels], vd2_m_c[NumberOfImageChannels];
				double vd1_c_c[NumberOfImageChannels], vd2_c_c[NumberOfImageChannels];
				float mask1_val, mask2_val;
				float mask_val;
				double mask_sum;
				//
				double corr;
				double corr_c[NumberOfImageChannels];
				BOOL corr_valid;
				float dcv_max, dcv_val;
				//
				for (d = 0; d < num_d; d++) {
					dcv[n][m][l][d] = 0;
				}
				//
				x1 = X1[n][m][l][0] + dX1[n][m][l][0];
				y1 = Y1[n][m][l][0] + dY1[n][m][l][0];
				z1 = Z1[n][m][l][0] + dZ1[n][m][l][0];
				//
				ys = 0;
				for (c = 0; c < d_num; c++) {
					vd1[c].GetAt(x1, y1, z1, &vd1_val);
					ys += vd1_val;
				}
				if (ys == 0) {
					continue;
				}
				//
				//*
				mask1.GetAt(x1, y1, z1, &mask1_val);
				if (mask1_val < 0.5) {
					continue;
				}
				//*/
				//
				x10 = x1 - ninv_cx;
				y10 = y1 - ninv_cy;
				z10 = z1 - ninv_cz;
				//
				for (k = 0; k < ninv_z; k++) {
					for (j = 0; j < ninv_y; j++) {
						for (i = 0; i < ninv_x; i++) {
							x1 = x10 + i;
							y1 = y10 + j;
							z1 = z10 + k;
							//
							if ((x1 <= 0) || (x1 >= vd_x-1) || (y1 <= 0) || (y1 >= vd_y-1) || (z1 <= 0) || (z1 >= vd_z-1)) {
								for (c = 0; c < d_num; c++) {
									vd1_patch[c][k][j][i] = 0;
								}
								//*
								vd1_patch_mask[k][j][i] = 0;
								/*/
								vd1_patch_mask[k][j][i] = 1;
								//*/
							} else {
								ix1 = (int)x1;
								iy1 = (int)y1;
								iz1 = (int)z1;
								//
								fx = x1 - ix1;
								fy = y1 - iy1;
								fz = z1 - iz1;
								//
								ys = 0;
								if (fx == 0 && fy == 0 && fz == 0) {
									for (c = 0; c < d_num; c++) {
										vd1_val = vd1[c].m_pData[iz1][iy1][ix1][0];
										vd1_patch[c][k][j][i] = vd1_val;
										ys += vd1_val;
									}
									vd1_patch_mask[k][j][i] = mask1.m_pData[iz1][iy1][ix1][0];
								} else {
									fx1 = 1.0 - fx;
									fy1 = 1.0 - fy;
									fz1 = 1.0 - fz;
									//
									for (c = 0; c < d_num; c++) {
										vd1_val  = fx1*fy1*fz1*vd1[c].m_pData[iz1  ][iy1  ][ix1  ][0];
										vd1_val += fx *fy1*fz1*vd1[c].m_pData[iz1  ][iy1  ][ix1+1][0];
										vd1_val += fx1*fy *fz1*vd1[c].m_pData[iz1  ][iy1+1][ix1  ][0];
										vd1_val += fx1*fy1*fz *vd1[c].m_pData[iz1+1][iy1  ][ix1  ][0];
										vd1_val += fx *fy *fz1*vd1[c].m_pData[iz1  ][iy1+1][ix1+1][0];
										vd1_val += fx *fy1*fz *vd1[c].m_pData[iz1+1][iy1  ][ix1+1][0];
										vd1_val += fx1*fy *fz *vd1[c].m_pData[iz1+1][iy1+1][ix1  ][0];
										vd1_val += fx *fy *fz *vd1[c].m_pData[iz1+1][iy1+1][ix1+1][0];
										vd1_patch[c][k][j][i] = vd1_val;
										ys += vd1_val;
									}
									//
									mask1_val  = fx1*fy1*fz1*mask1.m_pData[iz1  ][iy1  ][ix1  ][0];
									mask1_val += fx *fy1*fz1*mask1.m_pData[iz1  ][iy1  ][ix1+1][0];
									mask1_val += fx1*fy *fz1*mask1.m_pData[iz1  ][iy1+1][ix1  ][0];
									mask1_val += fx1*fy1*fz *mask1.m_pData[iz1+1][iy1  ][ix1  ][0];
									mask1_val += fx *fy *fz1*mask1.m_pData[iz1  ][iy1+1][ix1+1][0];
									mask1_val += fx *fy1*fz *mask1.m_pData[iz1+1][iy1  ][ix1+1][0];
									mask1_val += fx1*fy *fz *mask1.m_pData[iz1+1][iy1+1][ix1  ][0];
									mask1_val += fx *fy *fz *mask1.m_pData[iz1+1][iy1+1][ix1+1][0];
									vd1_patch_mask[k][j][i] = mask1_val;
								}
								//
								//*
								if (ys == 0) {
									vd1_patch_mask[k][j][i] = 0;
								}
								//*/
							}
						}
					}
				}
				//
				x20 = X2[n][m][l][0] + dX2[n][m][l][0] - ninv_cx;
				y20 = Y2[n][m][l][0] + dY2[n][m][l][0] - ninv_cy;
				z20 = Z2[n][m][l][0] + dZ2[n][m][l][0] - ninv_cz;
				//
				dcv_max = 0;
				for (d = 0; d < num_d; d++) {
					dx = disp_x[d];
					dy = disp_y[d];
					dz = disp_z[d];
					//
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								x2 = x20 + i + dx;
								y2 = y20 + j + dy;
								z2 = z20 + k + dz;
								//
								if ((x2 <= 0) || (x2 >= vd_x-1) || (y2 <= 0) || (y2 >= vd_y-1) || (z2 <= 0) || (z2 >= vd_z-1)) {
									for (c = 0; c < d_num; c++) {
										vd2_patch[c][k][j][i] = 0;
									}
									//*
									vd2_patch_mask[k][j][i] = 0;
									/*/
									vd2_patch_mask[k][j][i] = 1;
									//*/
								} else {
									ix2 = (int)x2;
									iy2 = (int)y2;
									iz2 = (int)z2;
									//
									fx = x2 - ix2;
									fy = y2 - iy2;
									fz = z2 - iz2;
									//
									fx1 = 1.0 - fx;
									fy1 = 1.0 - fy;
									fz1 = 1.0 - fz;
									//
									ys = 0;
									for (c = 0; c < d_num; c++) {
										vd2_val  = fx1*fy1*fz1*vd2[c].m_pData[iz2  ][iy2  ][ix2  ][0];
										vd2_val += fx *fy1*fz1*vd2[c].m_pData[iz2  ][iy2  ][ix2+1][0];
										vd2_val += fx1*fy *fz1*vd2[c].m_pData[iz2  ][iy2+1][ix2  ][0];
										vd2_val += fx1*fy1*fz *vd2[c].m_pData[iz2+1][iy2  ][ix2  ][0];
										vd2_val += fx *fy *fz1*vd2[c].m_pData[iz2  ][iy2+1][ix2+1][0];
										vd2_val += fx *fy1*fz *vd2[c].m_pData[iz2+1][iy2  ][ix2+1][0];
										vd2_val += fx1*fy *fz *vd2[c].m_pData[iz2+1][iy2+1][ix2  ][0];
										vd2_val += fx *fy *fz *vd2[c].m_pData[iz2+1][iy2+1][ix2+1][0];
										vd2_patch[c][k][j][i] = vd2_val;
										ys += vd2_val;
									}
									//
									mask2_val  = fx1*fy1*fz1*mask2.m_pData[iz2  ][iy2  ][ix2  ][0];
									mask2_val += fx *fy1*fz1*mask2.m_pData[iz2  ][iy2  ][ix2+1][0];
									mask2_val += fx1*fy *fz1*mask2.m_pData[iz2  ][iy2+1][ix2  ][0];
									mask2_val += fx1*fy1*fz *mask2.m_pData[iz2+1][iy2  ][ix2  ][0];
									mask2_val += fx *fy *fz1*mask2.m_pData[iz2  ][iy2+1][ix2+1][0];
									mask2_val += fx *fy1*fz *mask2.m_pData[iz2+1][iy2  ][ix2+1][0];
									mask2_val += fx1*fy *fz *mask2.m_pData[iz2+1][iy2+1][ix2  ][0];
									mask2_val += fx *fy *fz *mask2.m_pData[iz2+1][iy2+1][ix2+1][0];
									vd2_patch_mask[k][j][i] = mask2_val;
									//
									//*
									if (ys == 0) {
										vd2_patch_mask[k][j][i] = 0;
									}
									//*/
								}
							}
						}
					}
					//
					for (c = 0; c < d_num; c++) {
						vd1_m_c[c] = 0;
						vd2_m_c[c] = 0;
					}
					mask_sum = 0;
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								mask_val = vd1_patch_mask[k][j][i] * vd2_patch_mask[k][j][i];
								for (c = 0; c < d_num; c++) {
									vd1_m_c[c] += mask_val * vd1_patch[c][k][j][i];
									vd2_m_c[c] += mask_val * vd2_patch[c][k][j][i];
								}
								mask_sum += mask_val;
							}
						}
					}
					if (mask_sum != 0) {
						for (c = 0; c < d_num; c++) {
							vd1_m_c[c] /= mask_sum;
							vd2_m_c[c] /= mask_sum;
						}
					} else {
						dcv[n][m][l][d] = INVALID_VALUE;
						continue;
					}
					//
					corr = 0;
					corr_valid = TRUE;
					for (c = 0; c < d_num; c++) {
						vd1_c_c[c] = 0;
						vd2_c_c[c] = 0;
						corr_c[c]  = 0;
					}
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								double nm;
								ninv_val = ninv[k][j][i];
								mask_val = vd1_patch_mask[k][j][i] * vd2_patch_mask[k][j][i];
								nm = ninv_val * mask_val;
								//
								for (c = 0; c < d_num; c++) {
									vd1_val = vd1_patch[c][k][j][i] - vd1_m_c[c];
									vd2_val = vd2_patch[c][k][j][i] - vd2_m_c[c];
									//
									vd1_c_c[c] += nm * vd1_val * vd1_val;
									vd2_c_c[c] += nm * vd2_val * vd2_val;
									corr_c[c]  += nm * vd1_val * vd2_val;
								}
							}
						}
					}
					for (c = 0; c < d_num; c++) {
						if ((vd1_c_c[c] != 0) && (vd2_c_c[c] != 0)) {
							corr_c[c] = corr_c[c] / sqrt(vd1_c_c[c] * vd2_c_c[c]);
							/*
							if (corr_c[c] > 1.0) {
								corr_c[c] = 1.0;
							} else if (corr_c[c] < -1.0) {
								corr_c[c] = -1.0;
							}
							//*/
							corr += corr_c[c];
						} else {
							corr_valid = FALSE;
							break;
							//corr_c[c] = 0;
						}
					}
					if (!corr_valid) {
						dcv[n][m][l][d] = INVALID_VALUE;
					} else {
						dcv_val = (float)(255 * 128 * (1.0 - corr / d_num));
						//
						dcv[n][m][l][d] = dcv_val;
						if (dcv_val > dcv_max) {
							dcv_max = dcv_val;
						}
						//
						if (dcv_val >= 0) {
							HistMin = MIN(HistMin, dcv_val);
							HistMax = MAX(HistMax, dcv_val);
							total++;
						}
					}
				} // d
				//
				/*
				for (d = 0; d < num_d; d++) {
					if (dcv.m_pData[n][m][l][d] == INVALID_VALUE) {
						dcv.m_pData[n][m][l][d] = dcv_max;
					}
				}
				//*/
				//*
				BOOL have_invalid = FALSE;
				for (d = 0; d < num_d; d++) {
					if (dcv[n][m][l][d] == INVALID_VALUE) {
						have_invalid = TRUE;
						break;
					}
				}
				if (have_invalid) {
					for (d = 0; d < num_d; d++) {
						dcv[n][m][l][d] = 0;
					}
				}
				//*/
			} // l
		} // m
	} // n

	// compute the histogram info
	HistInterval = (double)(HistMax - HistMin) / nBins;

	for (n = 0; n < mesh_z; n++) {
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				int val;
				for (d = 0; d < num_d; d++) {
					if (dcv[n][m][l][d] > 0) {
						val = (int)MIN(dcv[n][m][l][d] / HistInterval, nBins-1);
						pHistogramBuffer[val]++;
					}
				}
			}
		}
	}

	// normalize the histogram
	for (ii = 0; ii < nBins; ii++) {
		pHistogramBuffer[ii] /= total;
	}

	// find the matching score
	Prob = 0;
	for (jj = 0; jj < 10; jj++) {
		MatchingScoreArr[jj] = -1;
	}
	for (ii = 0; ii < nBins; ii++) {
		Prob += pHistogramBuffer[ii];
		for (jj = 0; jj < 10; jj++) {
			if (Prob >= ProbArr[jj]) {
				if (MatchingScoreArr[jj] < 0) {
					MatchingScoreArr[jj] = MAX(ii, 1) * HistInterval+HistMin;
				}
			}
		}
	}
	TRACE2("Min: %f\n", HistMin);
	for (jj = 0; jj < 10; jj++) {
		TRACE2("%f, ", MatchingScoreArr[jj]);
	}
	TRACE2("\nMax: %f, total = %d\n", HistMax, total);

	/*
	for (n = 0; n < mesh_z; n++) {
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				for (d = 0; d < num_d; d++) {
					dcv.m_pData[n][m][l][d] = MIN(dcv.m_pData[n][m][l][d], DefaultMatchingScore);
				}
			}
		}
	}
	//*/

	for (k = 0; k < ninv_z; k++) {
		for (j = 0; j < ninv_y; j++) {
			free(ninv[k][j]);
		}
		free(ninv[k]);
	}
	free(ninv);
	//
	for (c = 0; c < d_num; c++) {
		for (k = 0; k < ninv_z; k++) {
			for (j = 0; j < ninv_y; j++) {
				free(vd1_patch[c][k][j]);
				free(vd2_patch[c][k][j]);
			}
			free(vd1_patch[c][k]);
			free(vd2_patch[c][k]);
		}
		free(vd1_patch[c]);
		free(vd2_patch[c]);
	}
	for (k = 0; k < ninv_z; k++) {
		for (j = 0; j < ninv_y; j++) {
			free(vd1_patch_mask[k][j]);
			free(vd2_patch_mask[k][j]);
		}
		free(vd1_patch_mask[k]);
		free(vd2_patch_mask[k]);
	}
	free(vd1_patch_mask);
	free(vd2_patch_mask);
	//
	delete pHistogramBuffer;

	return res;
}

BOOL ComputeDataCost3D_PP_CC(FVolume* vd1, FVolume* vd2, int d_num, FVolume& mask1, FVolume& mask2,
	REALV**** X1, REALV**** Y1, REALV**** Z1, REALV**** dX1, REALV**** dY1, REALV**** dZ1, REALV**** X2, REALV**** Y2, REALV**** Z2, REALV**** dX2, REALV**** dY2, REALV**** dZ2,
	int mesh_x, int mesh_y, int mesh_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV* disp_x, REALV* disp_y, REALV* disp_z, int num_d, REALV**** dcv,
	int dc_skip_back, float dc_back_color, int radius = 4, float weight = 1.0f)
{
	BOOL res = TRUE;
	//
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	//
	int vd_x, vd_y, vd_z;
	float vd_dx, vd_dy, vd_dz;
	//
	double*** vd1_patch[NumberOfImageChannels];
	double*** vd2_patch[NumberOfImageChannels];
	double*** vd1_patch_mask;
	double*** vd2_patch_mask;
	//const float INVALID_VALUE = -FLT_MAX;
	//
	int i, j, k, l, m, n, d, c;
	//
	/*
	double HistMin, HistMax;
	double HistInterval;
	double* pHistogramBuffer;
	int nBins = 20000;
	int total = 0; // total is the total number of plausible matches, used to normalize the histogram
	double Prob;
	double MatchingScoreArr[10];
	double ProbArr[10] = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
	int ii, jj;
	//*/

	// assume 4 channels for image
	if (d_num > NumberOfImageChannels) {
		return FALSE;
	}

	TRACE2("ComputeDataCost3D_PP_CC...\n");

	vd_x = vd1[0].m_vd_x;
	vd_y = vd1[0].m_vd_y;
	vd_z = vd1[0].m_vd_z;
	vd_dx = vd1[0].m_vd_dx;
	vd_dy = vd1[0].m_vd_dy;
	vd_dz = vd1[0].m_vd_dz;

	ninv_x = 2 * radius + 1;
	ninv_y = 2 * radius + 1;
	ninv_z = 2 * radius + 1;
	//
	ninv_cx = ninv_x / 2;
	ninv_cy = ninv_y / 2;
	ninv_cz = ninv_z / 2;

	for (c = 0; c < d_num; c++) {
		vd1_patch[c] = (double***)malloc(ninv_z * sizeof(double**));
		vd2_patch[c] = (double***)malloc(ninv_z * sizeof(double**));
		for (k = 0; k < ninv_z; k++) {
			vd1_patch[c][k] = (double**)malloc(ninv_y * sizeof(double*));
			vd2_patch[c][k] = (double**)malloc(ninv_y * sizeof(double*));
			for (j = 0; j < ninv_y; j++) {
				vd1_patch[c][k][j] = (double*)malloc(ninv_x * sizeof(double));
				vd2_patch[c][k][j] = (double*)malloc(ninv_x * sizeof(double));
			}
		}
	}
	vd1_patch_mask = (double***)malloc(ninv_z * sizeof(double**));
	vd2_patch_mask = (double***)malloc(ninv_z * sizeof(double**));
	for (k = 0; k < ninv_z; k++) {
		vd1_patch_mask[k] = (double**)malloc(ninv_y * sizeof(double*));
		vd2_patch_mask[k] = (double**)malloc(ninv_y * sizeof(double*));
		for (j = 0; j < ninv_y; j++) {
			vd1_patch_mask[k][j] = (double*)malloc(ninv_x * sizeof(double));
			vd2_patch_mask[k][j] = (double*)malloc(ninv_x * sizeof(double));
		}
	}

	/*
	pHistogramBuffer = new double[nBins];
	memset(pHistogramBuffer, 0, sizeof(double) * nBins);
	HistMin = 10e10;
	HistMax = 0;
	//*/

#if 0
	ComputeDataCost3D_CC(vd1[0].m_pData, vd1[0].m_vd_x, vd1[0].m_vd_y, vd1[0].m_vd_z, vd1[0].m_vd_s, vd2[0].m_pData, vd2[0].m_vd_x, vd2[0].m_vd_y, vd2[0].m_vd_z, vd2[0].m_vd_s, 
		X1, Y1, Z1, dX1, dY1, dZ1, X2, Y2, Z2, dX2, dY2, dZ2,
		mesh_x, mesh_y, mesh_z, mesh_ex, mesh_ey, mesh_ez, disp_x, disp_y, disp_z, num_d, dcv.m_pData,
		1, (float)0, 4);
#endif
#if 1
	double max_corr = -10000;
	double min_corr = 10000;
	for (n = 0; n < mesh_z; n++) {
		TRACE2("processing %d / %d total z\n", n, mesh_z);
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				float x20, y20, z20;
				float x2, y2, z2;
				float x10, y10, z10;
				float x1, y1, z1;
				float dx, dy, dz;
				float ys;
				int ix1, iy1, iz1;
				int ix2, iy2, iz2;
				double fx, fy, fz, fx1, fy1, fz1;
				//
				double vd1_val, vd2_val;
				double mask1_val, mask2_val;
				//
				double corr;
				double corr_c[NumberOfImageChannels];
				double corr_max = 1.0;
				//BOOL corr_valid;
				float dcv_max, dcv_val;
				//
				x1 = X1[n][m][l][0] + dX1[n][m][l][0];
				y1 = Y1[n][m][l][0] + dY1[n][m][l][0];
				z1 = Z1[n][m][l][0] + dZ1[n][m][l][0];
				//
				if ((x1 <= 0) || (x1 >= vd_x-1) || (y1 <= 0) || (y1 >= vd_y-1) || (z1 <= 0) || (z1 >= vd_z-1)) {
					for (d = 0; d < num_d; d++) {
						dcv[n][m][l][d] = 0;
					}
					continue;
				}
				//
				//*
				ys = 0;
				for (c = 0; c < d_num; c++) {
					float val;
					vd1[c].GetAt(x1, y1, z1, &val);
					ys += val;
				}
				if (ys == 0) {
					for (d = 0; d < num_d; d++) {
						dcv[n][m][l][d] = 0;
					}
					continue;
				}
				//*/
				//
				/*
				mask1.GetAt(x1, y1, z1, &mask1_val);
				if (mask1_val < 0.5) {
					for (d = 0; d < num_d; d++) {
						dcv[n][m][l][d] = 0;
					}
					continue;
				}
				//*/
				//
				x10 = x1 - ninv_cx;
				y10 = y1 - ninv_cy;
				z10 = z1 - ninv_cz;
				//
				for (k = 0; k < ninv_z; k++) {
					for (j = 0; j < ninv_y; j++) {
						for (i = 0; i < ninv_x; i++) {
							x1 = x10 + i;
							y1 = y10 + j;
							z1 = z10 + k;
							//
							if ((x1 <= 0) || (x1 >= vd_x-1) || (y1 <= 0) || (y1 >= vd_y-1) || (z1 <= 0) || (z1 >= vd_z-1)) {
								for (c = 0; c < d_num; c++) {
									vd1_patch[c][k][j][i] = 0;
								}
								/*
								vd1_patch_mask[k][j][i] = 0;
								/*/
								vd1_patch_mask[k][j][i] = 1;
								//*/
							} else {
								ix1 = (int)x1;
								iy1 = (int)y1;
								iz1 = (int)z1;
								fx = x1 - ix1;
								fy = y1 - iy1;
								fz = z1 - iz1;
								//
								ys = 0;
								if (fx == 0 && fy == 0 && fz == 0) {
									for (c = 0; c < d_num; c++) {
										vd1_val = vd1[c].m_pData[iz1][iy1][ix1][0];
										vd1_patch[c][k][j][i] = vd1_val;
										ys += vd1_val;
									}
									vd1_patch_mask[k][j][i] = mask1.m_pData[iz1][iy1][ix1][0];
								} else {
									fx1 = 1.0f - fx;
									fy1 = 1.0f - fy;
									fz1 = 1.0f - fz;
									//
									for (c = 0; c < d_num; c++) {
										vd1_val  = fx1*fy1*fz1*vd1[c].m_pData[iz1  ][iy1  ][ix1  ][0];
										vd1_val += fx *fy1*fz1*vd1[c].m_pData[iz1  ][iy1  ][ix1+1][0];
										vd1_val += fx1*fy *fz1*vd1[c].m_pData[iz1  ][iy1+1][ix1  ][0];
										vd1_val += fx1*fy1*fz *vd1[c].m_pData[iz1+1][iy1  ][ix1  ][0];
										vd1_val += fx *fy *fz1*vd1[c].m_pData[iz1  ][iy1+1][ix1+1][0];
										vd1_val += fx *fy1*fz *vd1[c].m_pData[iz1+1][iy1  ][ix1+1][0];
										vd1_val += fx1*fy *fz *vd1[c].m_pData[iz1+1][iy1+1][ix1  ][0];
										vd1_val += fx *fy *fz *vd1[c].m_pData[iz1+1][iy1+1][ix1+1][0];
										vd1_patch[c][k][j][i] = vd1_val;
										ys += vd1_val;
									}
									//
									mask1_val  = fx1*fy1*fz1*mask1.m_pData[iz1  ][iy1  ][ix1  ][0];
									mask1_val += fx *fy1*fz1*mask1.m_pData[iz1  ][iy1  ][ix1+1][0];
									mask1_val += fx1*fy *fz1*mask1.m_pData[iz1  ][iy1+1][ix1  ][0];
									mask1_val += fx1*fy1*fz *mask1.m_pData[iz1+1][iy1  ][ix1  ][0];
									mask1_val += fx *fy *fz1*mask1.m_pData[iz1  ][iy1+1][ix1+1][0];
									mask1_val += fx *fy1*fz *mask1.m_pData[iz1+1][iy1  ][ix1+1][0];
									mask1_val += fx1*fy *fz *mask1.m_pData[iz1+1][iy1+1][ix1  ][0];
									mask1_val += fx *fy *fz *mask1.m_pData[iz1+1][iy1+1][ix1+1][0];
									vd1_patch_mask[k][j][i] = mask1_val;
								}
								//
								/*
								if (ys == 0) {
									vd1_patch_mask[k][j][i] = 0;
								}
								//*/
							}
						}
					}
				}
				//
				x20 = X2[n][m][l][0] + dX2[n][m][l][0] - ninv_cx;
				y20 = Y2[n][m][l][0] + dY2[n][m][l][0] - ninv_cy;
				z20 = Z2[n][m][l][0] + dZ2[n][m][l][0] - ninv_cz;
				//
				dcv_max = 0;
				for (d = 0; d < num_d; d++) {
					dx = disp_x[d];
					dy = disp_y[d];
					dz = disp_z[d];
					//
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								x2 = x20 + i + dx;
								y2 = y20 + j + dy;
								z2 = z20 + k + dz;
								//
								if ((x2 <= 0) || (x2 >= vd_x-1) || (y2 <= 0) || (y2 >= vd_y-1) || (z2 <= 0) || (z2 >= vd_z-1)) {
									for (c = 0; c < d_num; c++) {
										vd2_patch[c][k][j][i] = 0;
									}
									/*
									vd2_patch_mask[k][j][i] = 0;
									/*/
									vd2_patch_mask[k][j][i] = 1;
									//*/
								} else {
									ix2 = (int)x2;
									iy2 = (int)y2;
									iz2 = (int)z2;
									fx = x2 - ix2;
									fy = y2 - iy2;
									fz = z2 - iz2;
									//
									ys = 0;
									if (fx == 0 && fy == 0 && fz == 0) {
										for (c = 0; c < d_num; c++) {
											vd2_val = vd2[c].m_pData[iz2][iy2][ix2][0];
											vd2_patch[c][k][j][i] = vd2_val;
											ys += vd2_val;
										}
										vd2_patch_mask[k][j][i] = mask2.m_pData[iz2][iy2][ix2][0];
									} else {
										fx1 = 1.0f - fx;
										fy1 = 1.0f - fy;
										fz1 = 1.0f - fz;
										//
										for (c = 0; c < d_num; c++) {
											vd2_val  = fx1*fy1*fz1*vd2[c].m_pData[iz2  ][iy2  ][ix2  ][0];
											vd2_val += fx *fy1*fz1*vd2[c].m_pData[iz2  ][iy2  ][ix2+1][0];
											vd2_val += fx1*fy *fz1*vd2[c].m_pData[iz2  ][iy2+1][ix2  ][0];
											vd2_val += fx1*fy1*fz *vd2[c].m_pData[iz2+1][iy2  ][ix2  ][0];
											vd2_val += fx *fy *fz1*vd2[c].m_pData[iz2  ][iy2+1][ix2+1][0];
											vd2_val += fx *fy1*fz *vd2[c].m_pData[iz2+1][iy2  ][ix2+1][0];
											vd2_val += fx1*fy *fz *vd2[c].m_pData[iz2+1][iy2+1][ix2  ][0];
											vd2_val += fx *fy *fz *vd2[c].m_pData[iz2+1][iy2+1][ix2+1][0];
											vd2_patch[c][k][j][i] = vd2_val;
											ys += vd2_val;
										}
										//
										mask2_val  = fx1*fy1*fz1*mask2.m_pData[iz2  ][iy2  ][ix2  ][0];
										mask2_val += fx *fy1*fz1*mask2.m_pData[iz2  ][iy2  ][ix2+1][0];
										mask2_val += fx1*fy *fz1*mask2.m_pData[iz2  ][iy2+1][ix2  ][0];
										mask2_val += fx1*fy1*fz *mask2.m_pData[iz2+1][iy2  ][ix2  ][0];
										mask2_val += fx *fy *fz1*mask2.m_pData[iz2  ][iy2+1][ix2+1][0];
										mask2_val += fx *fy1*fz *mask2.m_pData[iz2+1][iy2  ][ix2+1][0];
										mask2_val += fx1*fy *fz *mask2.m_pData[iz2+1][iy2+1][ix2  ][0];
										mask2_val += fx *fy *fz *mask2.m_pData[iz2+1][iy2+1][ix2+1][0];
										vd2_patch_mask[k][j][i] = mask2_val;
									}
									//
									/*
									if (ys == 0) {
										vd2_patch_mask[k][j][i] = 0;
									}
									//*/
								}
							}
						}
					}
					//
#if 0
					{
						double fixedMean[NumberOfImageChannels], movingMean[NumberOfImageChannels];
						double suma2[NumberOfImageChannels], sumb2[NumberOfImageChannels], suma[NumberOfImageChannels], sumb[NumberOfImageChannels], sumab[NumberOfImageChannels];
						double sff[NumberOfImageChannels], smm[NumberOfImageChannels], sfm[NumberOfImageChannels];
						double count;

						for (c = 0; c < d_num; c++) {
							suma2[c] = suma[c] = 0;
							sumb2[c] = sumb[c] = 0;
							sumab[c] = 0;
						}
						count = 0;
						for (k = 0; k < ninv_z; k++) {
							for (j = 0; j < ninv_y; j++) {
								for (i = 0; i < ninv_x; i++) {
									for (c = 0; c < d_num; c++) {
										vd1_val = vd1_patch[c][k][j][i];
										vd2_val = vd2_patch[c][k][j][i];
										//
										suma[c]  += vd1_val;
										suma2[c] += vd1_val * vd1_val;
										sumb[c]  += vd2_val;
										sumb2[c] += vd2_val * vd2_val;
										sumab[c] += vd1_val * vd2_val;
									}
									count += 1;
								}
							}
						}
						if (count > 0) {
							corr = 0;
							//corr_valid = TRUE;
							for (c = 0; c < d_num; c++) {
								fixedMean[c]  = suma[c] / count;
								movingMean[c] = sumb[c] / count;
								sff[c] = suma2[c] - fixedMean[c]*suma[c]  - fixedMean[c]*suma[c]  + count*fixedMean[c]*fixedMean[c];
								smm[c] = sumb2[c] - movingMean[c]*sumb[c] - movingMean[c]*sumb[c] + count*movingMean[c]*movingMean[c];
								sfm[c] = sumab[c] - movingMean[c]*suma[c] - fixedMean[c]*sumb[c]  + count*movingMean[c]*fixedMean[c];
								//
								if ((sff[c] > 0) && (smm[c] > 0)) {
#ifdef USE_CC_NCC
									corr_c[c] = sfm[c] / sqrt(sff[c] * smm[c]);
#else
									corr_c[c] = sfm[c]*sfm[c] / (sff[c] * smm[c]);
#endif
									if (corr_c[c] > corr_max) {
										corr_c[c] = corr_max;
									} else if (corr_c[c] < corr_min) {
										corr_c[c] = corr_min;
									}
									corr += corr_c[c];
								} else {
									corr_c[c] = 0;
									//corr_valid = FALSE;
									//break;
								}

							}
						} else {
							//dcv[n][m][l][d] = INVALID_VALUE;
							//continue;

							corr = 0;
							for (c = 0; c < d_num; c++) {
								corr_c[c] = 0;
							}
						}
					}
#endif
#if 1
					{
						double fixedMean[NumberOfImageChannels], movingMean[NumberOfImageChannels];
						double sff[NumberOfImageChannels], smm[NumberOfImageChannels], sfm[NumberOfImageChannels];
						double sffw, smmw, sfmw;
						double mask1_sum, mask2_sum;
						double mask1_val2, mask2_val2, mask12_val;

						for (c = 0; c < d_num; c++) {
							fixedMean[c] = 0;
							movingMean[c] = 0;
						}
						mask1_sum = 0;
						mask2_sum = 0;
						for (k = 0; k < ninv_z; k++) {
							for (j = 0; j < ninv_y; j++) {
								for (i = 0; i < ninv_x; i++) {
									/*
									mask1_val = vd1_patch_mask[k][j][i];
									mask2_val = vd2_patch_mask[k][j][i];
									/*/
									mask1_val = 1;
									mask2_val = 1;
									//*/
									//
									for (c = 0; c < d_num; c++) {
										fixedMean[c]  += mask1_val * vd1_patch[c][k][j][i];
										movingMean[c] += mask2_val * vd2_patch[c][k][j][i];
									}
									//
									mask1_sum += mask1_val;
									mask2_sum += mask2_val;
								}
							}
						}
						if (mask1_sum > 0 && mask2_sum > 0) {
							corr = 0;
							//corr_valid = TRUE;
							//
							for (c = 0; c < d_num; c++) {
								fixedMean[c]  /= mask1_sum;
								movingMean[c] /= mask2_sum;
								//
								sff[c] = 0;
								smm[c] = 0;
								sfm[c] = 0;
							}
							sffw = 0;
							smmw = 0;
							sfmw = 0;
							for (k = 0; k < ninv_z; k++) {
								for (j = 0; j < ninv_y; j++) {
									for (i = 0; i < ninv_x; i++) {
										/*
										mask1_val = vd1_patch_mask[k][j][i];
										mask2_val = vd2_patch_mask[k][j][i];
										/*/
										mask1_val = 1;
										mask2_val = 1;
										//*/
										mask1_val2 = mask1_val * mask1_val;
										mask2_val2 = mask2_val * mask2_val;
										mask12_val = mask1_val * mask2_val;
										//
										for (c = 0; c < d_num; c++) {
											vd1_val = vd1_patch[c][k][j][i] - fixedMean[c];
											vd2_val = vd2_patch[c][k][j][i] - movingMean[c];
											//
											sff[c] += mask1_val2 * vd1_val * vd1_val;
											smm[c] += mask2_val2 * vd2_val * vd2_val;
											sfm[c] += mask12_val * vd1_val * vd2_val;
										}
										//
										sffw += mask1_val2;
										smmw += mask2_val2;
										sfmw += mask12_val;
									}
								}
							}
							for (c = 0; c < d_num; c++) {
								//if ((sff[c] > 0) && (smm[c] > 0) && (sfmw > 0)) {
								if ((sff[c] > 1e-1) && (smm[c] > 1e-1) && (sfmw > 0)) {
									sfm[c] /= sfmw;
									sff[c] /= sffw;
									smm[c] /= smmw;
#ifdef USE_CC_NCC
									corr_c[c] = sfm[c] / sqrt(sff[c] * smm[c]);
#else
									corr_c[c] = (sfm[c] * sfm[c]) / (sff[c] * smm[c]);
#endif

									if (corr_c[c] > max_corr) {
										max_corr = corr_c[c];
									}
									if (corr_c[c] < min_corr) {
										min_corr = corr_c[c];
									}

									//*
									if (corr_c[c] > corr_max) {
										corr_c[c] = corr_max;
									}
									//*/

									corr += corr_c[c];
								} else {
									corr_c[c] = 0;
									//corr_valid = FALSE;
									//break;
								}
							}
						} else {
							//dcv[n][m][l][d] = INVALID_VALUE;
							//continue;

							corr = 0;
							for (c = 0; c < d_num; c++) {
								corr_c[c] = 0;
							}
						}
					}
#endif
					//
					//if (!corr_valid) {
					//	dcv[n][m][l][d] = INVALID_VALUE;
					//} else {
						dcv_val = weight * (float)(255 * 128 * (corr_max - corr / d_num));
						//
						dcv[n][m][l][d] = dcv_val;
						/*
						if (dcv_val > dcv_max) {
							dcv_max = dcv_val;
						}
						//
						if (dcv_val >= 0) {
							HistMin = MIN(HistMin, dcv_val);
							HistMax = MAX(HistMax, dcv_val);
							total++;
						}
						//*/
					//}
				} // d
				//
				/*
				for (d = 0; d < num_d; d++) {
					if (dcv[n][m][l][d] == INVALID_VALUE) {
						dcv[n][m][l][d] = dcv_max;
					}
				}
				//*/
				/*
				BOOL have_invalid = FALSE;
				for (d = 0; d < num_d; d++) {
					if (dcv[n][m][l][d] == INVALID_VALUE) {
						have_invalid = TRUE;
						break;
					}
				}
				if (have_invalid) {
					for (d = 0; d < num_d; d++) {
						dcv[n][m][l][d] = 0;
					}
				}
				//*/
			} // l
		} // m
	} // n

	TRACE2("\nmin_corr: %f, max_corr = %f\n", min_corr, max_corr);

#if 0
	// compute the histogram info
	HistInterval = (double)(HistMax - HistMin) / nBins;

	for (n = 0; n < mesh_z; n++) {
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				int val;
				for (d = 0; d < num_d; d++) {
					if (dcv.m_pData[n][m][l][d] > 0) {
						val = MIN(dcv.m_pData[n][m][l][d] / HistInterval, nBins-1);
						pHistogramBuffer[val]++;
					}
				}
			}
		}
	}

	// normalize the histogram
	for (ii = 0; ii < nBins; ii++) {
		pHistogramBuffer[ii] /= total;
	}

	// find the matching score
	Prob = 0;
	for (jj = 0; jj < 10; jj++) {
		MatchingScoreArr[jj] = -1;
	}
	for (ii = 0; ii < nBins; ii++) {
		Prob += pHistogramBuffer[ii];
		for (jj = 0; jj < 10; jj++) {
			if (Prob >= ProbArr[jj]) {
				if (MatchingScoreArr[jj] < 0) {
					MatchingScoreArr[jj] = MAX(ii, 1) * HistInterval+HistMin;
				}
			}
		}
	}
	TRACE2("Min: %f\n", HistMin);
	for (jj = 0; jj < 10; jj++) {
		TRACE2("%f, ", MatchingScoreArr[jj]);
	}
	TRACE2("\nMax: %f, total = %d\n", HistMax, total);

	/*
	for (n = 0; n < mesh_z; n++) {
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				for (d = 0; d < num_d; d++) {
					dcv.m_pData[n][m][l][d] = MIN(dcv.m_pData[n][m][l][d], DefaultMatchingScore);
				}
			}
		}
	}
	//*/
#endif
#endif

	for (c = 0; c < d_num; c++) {
		for (k = 0; k < ninv_z; k++) {
			for (j = 0; j < ninv_y; j++) {
				free(vd1_patch[c][k][j]);
				free(vd2_patch[c][k][j]);
			}
			free(vd1_patch[c][k]);
			free(vd2_patch[c][k]);
		}
		free(vd1_patch[c]);
		free(vd2_patch[c]);
	}
	for (k = 0; k < ninv_z; k++) {
		for (j = 0; j < ninv_y; j++) {
			free(vd1_patch_mask[k][j]);
			free(vd2_patch_mask[k][j]);
		}
		free(vd1_patch_mask[k]);
		free(vd2_patch_mask[k]);
	}
	free(vd1_patch_mask);
	free(vd2_patch_mask);
	//
	//delete pHistogramBuffer;

	return res;
}

BOOL ComputeDataCost3D_PP_CC_Fast(FVolume* vd1, FVolume* vd2, int d_num, FVolume& mask1, FVolume& mask2,
	REALV**** XC, REALV**** YC, REALV**** ZC,
	int mesh_x, int mesh_y, int mesh_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV* disp_x, REALV* disp_y, REALV* disp_z, int num_d, REALV**** dcv,
	int dc_skip_back, float dc_back_color, int radius = 4, float weight = 1.0f)
{
	BOOL res = TRUE;
	//
	int vd1_x, vd1_y, vd1_z;
	int vd2_x, vd2_y, vd2_z;
	float vd1_dx, vd1_dy, vd1_dz;
	float vd2_dx, vd2_dy, vd2_dz;
	//
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	int ninv_size;
	int i, j, k, l, m, n, d, c;
	double max_corr, min_corr;
	REALV disp_x_max, disp_x_min, disp_y_max, disp_y_min, disp_z_max, disp_z_min;
	int lx, ly, lz, rx, ry, rz;
	FVolume vd1t[NumberOfImageChannels], vd2t[NumberOfImageChannels];
	FVolume mask1t, mask2t;
	int vdt_x, vdt_y, vdt_z;
	//const float INVALID_VALUE = -FLT_MAX;

	// assume 4 channels for image
	if (d_num > NumberOfImageChannels) {
		return FALSE;
	}

	TRACE2("ComputeDataCost3D_PP_CC_Fast...\n");

	vd1_x = vd1[0].m_vd_x; vd1_y = vd1[0].m_vd_y; vd1_z = vd1[0].m_vd_z;
	vd2_x = vd2[0].m_vd_x; vd2_y = vd2[0].m_vd_y; vd2_z = vd2[0].m_vd_z;
	vd1_dx = vd1[0].m_vd_dx; vd1_dy = vd1[0].m_vd_dy; vd1_dz = vd1[0].m_vd_dz;
	vd2_dx = vd2[0].m_vd_dx; vd2_dy = vd2[0].m_vd_dy; vd2_dz = vd2[0].m_vd_dz;

	ninv_x = 2 * radius + 1;
	ninv_y = 2 * radius + 1;
	ninv_z = 2 * radius + 1;
	ninv_size = ninv_x * ninv_y * ninv_z;
	//
	ninv_cx = ninv_x / 2;
	ninv_cy = ninv_y / 2;
	ninv_cz = ninv_z / 2;

	disp_x_max = -10000; disp_x_min = 10000;
	disp_y_max = -10000; disp_y_min = 10000;
	disp_z_max = -10000; disp_z_min = 10000;
	for (d = 0; d < num_d; d++) {
		if (disp_x[d] > disp_x_max) { disp_x_max = disp_x[d]; }
		if (disp_x[d] < disp_x_min) { disp_x_min = disp_x[d]; }
		if (disp_y[d] > disp_y_max) { disp_y_max = disp_x[d]; }
		if (disp_y[d] < disp_y_min) { disp_y_min = disp_x[d]; }
		if (disp_z[d] > disp_z_max) { disp_z_max = disp_x[d]; }
		if (disp_z[d] < disp_z_min) { disp_z_min = disp_x[d]; }
	}
	
	lx = (int)(fabs(disp_x_min) + 0.999999) + ninv_cx;
	rx = (int)(fabs(disp_x_max) + 0.999999) + ninv_cx+1;
	ly = (int)(fabs(disp_y_min) + 0.999999) + ninv_cy;
	ry = (int)(fabs(disp_y_max) + 0.999999) + ninv_cy+1;
	lz = (int)(fabs(disp_z_min) + 0.999999) + ninv_cz;
	rz = (int)(fabs(disp_z_max) + 0.999999) + ninv_cz+1;

	vdt_x = max(vd1_x+lx+rx, vd2_x+lx+rx);
	vdt_y = max(vd1_y+ly+ry, vd2_y+ly+ry);
	vdt_z = max(vd1_z+lz+rz, vd2_z+lz+rz);

	for (c = 0; c < d_num; c++) {
		vd1t[c].allocate(vdt_x, vdt_y, vdt_z);
		vd2t[c].allocate(vdt_x, vdt_y, vdt_z);
	}
	mask1t.allocate(vdt_x, vdt_y, vdt_z);
	mask2t.allocate(vdt_x, vdt_y, vdt_z);
	
#if 0
	ComputeDataCost3D_CC_Fast(vd1[0].m_pData, vd1[0].m_vd_x, vd1[0].m_vd_y, vd1[0].m_vd_z, vd1[0].m_vd_s, vd2[0].m_pData, vd2[0].m_vd_x, vd2[0].m_vd_y, vd2[0].m_vd_z, vd2[0].m_vd_s, 
		XC, YC, ZC,
		mesh_x, mesh_y, mesh_z, mesh_ex, mesh_ey, mesh_ez, 
		disp_x, disp_y, disp_z, num_d, dcv.m_pData,
		1, (float)0, 4);
#endif
#if 1
	max_corr = -10000;
	min_corr = 10000;
	for (d = 0; d < num_d; d++) {
		float xc, yc, zc;
		float x2, y2, z2;
		float dx, dy, dz;
		float ys;
		int ix1, iy1, iz1;
		int ix2, iy2, iz2;
		int ixc, iyc, izc;
		int ix, iy, iz;
		double vd1_val, vd2_val;
		double mask1_val, mask2_val;
		double fx, fy, fz, fx1, fy1, fz1;
		double fixedMean, movingMean;
		double suma2[NumberOfImageChannels], sumb2[NumberOfImageChannels], suma[NumberOfImageChannels], sumb[NumberOfImageChannels], sumab[NumberOfImageChannels], count;
		double sff, smm, sfm;
		double corr, corr_c[NumberOfImageChannels];
		double corr_max = 1.0;

		if (d % (int)(num_d * 0.1) == 0) {
			TRACE2("processing %d / %d total d\n", d, num_d);
		}

		dx = disp_x[d];
		dy = disp_y[d];
		dz = disp_z[d];

		// translate images
		#pragma omp parallel for private(i,j,k,ix1,iy1,iz1,c,x2,y2,z2,ix2,iy2,iz2,fx,fy,fz,fx1,fy1,fz1,vd2_val,mask2_val)
		for (k = 0; k < vdt_z; k++) {
			for (j = 0; j < vdt_y; j++) {
				for (i = 0; i < vdt_x; i++) {
					ix1 = i - lx;
					iy1 = j - ly;
					iz1 = k - lz;
					if ((ix1 < 0) || (ix1 > vd1_x-1) || (iy1 < 0) || (iy1 > vd1_y-1) || (iz1 < 0) || (iz1 > vd1_z-1)) {
						for (c = 0; c < d_num; c++) {
							vd1t[c].m_pData[k][j][i][0] = 0;
						}
						mask1t.m_pData[k][j][i][0] = 0;
					} else {
						for (c = 0; c < d_num; c++) {
							vd1t[c].m_pData[k][j][i][0] = vd1[c].m_pData[iz1][iy1][ix1][0];
						}
						mask1t.m_pData[k][j][i][0] = mask1.m_pData[iz1][iy1][ix1][0];
					}

					x2 = i - (lx - dx);
					y2 = j - (ly - dy);
					z2 = k - (lz - dz);
					ix2 = (int)x2;
					iy2 = (int)y2;
					iz2 = (int)z2;
					fx = x2 - ix2;
					fy = y2 - iy2;
					fz = z2 - iz2;
					if (fx == 0 && fy == 0 && fz == 0) {
						if ((ix2 < 0) || (ix2 > vd2_x-1) || (iy2 < 0) || (iy2 > vd2_y-1) || (iz2 < 0) || (iz2 > vd2_z-1)) {
							for (c = 0; c < d_num; c++) {
								vd2t[c].m_pData[k][j][i][0] = 0;
							}
							mask2t.m_pData[k][j][i][0] = 0;
						} else {
							for (c = 0; c < d_num; c++) {
								vd2t[c].m_pData[k][j][i][0] = vd2[c].m_pData[iz2][iy2][ix2][0];
							}
							mask2t.m_pData[k][j][i][0] = mask2.m_pData[iz2][iy2][ix2][0];
						}
					} else {
						if ((ix2 < 0) || (ix2 >= vd2_x-1) || (iy2 < 0) || (iy2 >= vd2_y-1) || (iz2 < 0) || (iz2 >= vd2_z-1)) {
							for (c = 0; c < d_num; c++) {
								vd2t[c].m_pData[k][j][i][0] = 0;
							}
							mask2t.m_pData[k][j][i][0] = 0;
						} else {
							fx1 = 1.0f - fx;
							fy1 = 1.0f - fy;
							fz1 = 1.0f - fz;
							for (c = 0; c < d_num; c++) {
								vd2_val  = fx1*fy1*fz1*vd2[c].m_pData[iz2  ][iy2  ][ix2  ][0];
								vd2_val += fx *fy1*fz1*vd2[c].m_pData[iz2  ][iy2  ][ix2+1][0];
								vd2_val += fx1*fy *fz1*vd2[c].m_pData[iz2  ][iy2+1][ix2  ][0];
								vd2_val += fx1*fy1*fz *vd2[c].m_pData[iz2+1][iy2  ][ix2  ][0];
								vd2_val += fx *fy *fz1*vd2[c].m_pData[iz2  ][iy2+1][ix2+1][0];
								vd2_val += fx *fy1*fz *vd2[c].m_pData[iz2+1][iy2  ][ix2+1][0];
								vd2_val += fx1*fy *fz *vd2[c].m_pData[iz2+1][iy2+1][ix2  ][0];
								vd2_val += fx *fy *fz *vd2[c].m_pData[iz2+1][iy2+1][ix2+1][0];
								vd2t[c].m_pData[k][j][i][0] = vd2_val;
							}
							//
							mask2_val  = fx1*fy1*fz1*mask2.m_pData[iz2  ][iy2  ][ix2  ][0];
							mask2_val += fx *fy1*fz1*mask2.m_pData[iz2  ][iy2  ][ix2+1][0];
							mask2_val += fx1*fy *fz1*mask2.m_pData[iz2  ][iy2+1][ix2  ][0];
							mask2_val += fx1*fy1*fz *mask2.m_pData[iz2+1][iy2  ][ix2  ][0];
							mask2_val += fx *fy *fz1*mask2.m_pData[iz2  ][iy2+1][ix2+1][0];
							mask2_val += fx *fy1*fz *mask2.m_pData[iz2+1][iy2  ][ix2+1][0];
							mask2_val += fx1*fy *fz *mask2.m_pData[iz2+1][iy2+1][ix2  ][0];
							mask2_val += fx *fy *fz *mask2.m_pData[iz2+1][iy2+1][ix2+1][0];
							mask2t.m_pData[k][j][i][0] = mask2_val;
						}
					}
				}
			}
		}
		/*
		for (c = 0; c < d_num; c++) {
			char filename[1024];
			sprintf(filename, "vd1t_%d.nii.gz", c);
			vd1t[c].save(filename, 1);
			sprintf(filename, "vd2t_%d.nii.gz", c);
			vd2t[c].save(filename, 1);
		}
		mask1t.save("mask1t.nii.gz", 1);
		mask2t.save("mask2t.nii.gz", 1);
		//*/

		#pragma omp parallel for private(l,m,n,xc,yc,zc,ys,c,ixc,iyc,izc,count,suma,suma2,sumb,sumb2,sumab,i,j,k,ix,iy,iz,mask1_val,mask2_val,vd1_val,vd2_val,fixedMean,movingMean,sff,smm,sfm,corr_c,corr) shared(max_corr,min_corr)
		for (n = 0; n < mesh_z; n++) {
			for (m = 0; m < mesh_y; m++) {
				for (l = 0; l < mesh_x; l++) {
					xc = XC[n][m][l][0];
					yc = YC[n][m][l][0];
					zc = ZC[n][m][l][0];
					//
					if (dc_skip_back == 1) {
						if ((xc < 0) || (xc > vd1_x-1) || (yc < 0) || (yc > vd1_y-1) || (zc < 0) || (zc > vd1_z-1)) {
							dcv[n][m][l][d] = 0;
							continue;
						} else {
							//*
							ys = 0;
							for (c = 0; c < d_num; c++) {
								ys += vd1[c].m_pData[(int)zc][(int)yc][(int)xc][0];
							}
							if (ys <= dc_back_color) {
								dcv[n][m][l][d] = 0;
								continue;
							}
							//*/
						}
					}
					//
					ixc = (int)(xc - ninv_cx + lx);
					iyc = (int)(yc - ninv_cy + ly);
					izc = (int)(zc - ninv_cz + lz);
					//
					count = 0;
					for (c = 0; c < d_num; c++) {
						suma2[c] = suma[c] = 0;
						sumb2[c] = sumb[c] = 0;
						sumab[c] = 0;
					}
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								ix = ixc + i;
								iy = iyc + j;
								iz = izc + k;
								//
								mask1_val = mask1t.m_pData[iz][iy][ix][0];
								mask2_val = mask2t.m_pData[iz][iy][ix][0];
								if (mask1_val > 0 && mask2_val > 0)
								{
									for (c = 0; c < d_num; c++) {
										vd1_val = vd1t[c].m_pData[iz][iy][ix][0];
										vd2_val = vd2t[c].m_pData[iz][iy][ix][0];
										//
										suma[c]  += vd1_val;
										suma2[c] += vd1_val * vd1_val;
										sumb[c]  += vd2_val;
										sumb2[c] += vd2_val * vd2_val;
										sumab[c] += vd1_val * vd2_val;
									}
									count += 1;
								}
							}
						}
					}
					//
					corr = 0;
					if (count > 0) {
						for (c = 0; c < d_num; c++) {
							fixedMean  = suma[c] / count;
							movingMean = sumb[c] / count;
							sff = suma2[c] -  fixedMean*suma[c] -  fixedMean*suma[c] + count* fixedMean* fixedMean;
							smm = sumb2[c] - movingMean*sumb[c] - movingMean*sumb[c] + count*movingMean*movingMean;
							sfm = sumab[c] - movingMean*suma[c] -  fixedMean*sumb[c] + count*movingMean* fixedMean;
							//
							//if ((sff > 0) && (smm > 0)) {
							//if ((sff > 1e-1) && (smm > 1e-1)) {
							if (sff*smm > 1.e-5) {
#ifdef USE_CC_NCC
								corr_c[c] = sfm / sqrt(sff * smm);
#else
								corr_c[c] = (sfm * sfm) / (sff * smm);
#endif

								#pragma omp critical
								{
									if (corr_c[c] > max_corr) {
										max_corr = corr_c[c];
									}
									if (corr_c[c] < min_corr) {
										min_corr = corr_c[c];
									}
								}
								//*
								if (corr_c[c] > corr_max) {
									corr_c[c] = corr_max;
								}
								//*/
								corr += corr_c[c];
							} else {
								corr_c[c] = corr_max;
							}
						}
						corr /= d_num;
					} else {
						corr = corr_max;
					}
					//
					dcv[n][m][l][d] = weight * (float)(255 * 128 * (corr_max - corr));
				}
			}
		}
	}

	TRACE2("\nmin_corr: %f, max_corr = %f\n", min_corr, max_corr);
#endif

	return res;
}

BOOL ComputeDataCost3D_PP_NMI(FVolume* vd1, FVolume* vd2, int d_num, FVolume& mask1, FVolume& mask2,
	REALV**** X1, REALV**** Y1, REALV**** Z1, REALV**** dX1, REALV**** dY1, REALV**** dZ1, REALV**** X2, REALV**** Y2, REALV**** Z2, REALV**** dX2, REALV**** dY2, REALV**** dZ2,
	int mesh_x, int mesh_y, int mesh_z, int mesh_ex, int mesh_ey, int mesh_ez,
	REALV* disp_x, REALV* disp_y, REALV* disp_z, int num_d, REALV**** dcv,
	int dc_weight, int dc_skip_back, float dc_back_color, int ninv_s = 1)
{
	BOOL res = TRUE;
	//
	float*** ninv;
	int ninv_x, ninv_y, ninv_z;
	int ninv_cx, ninv_cy, ninv_cz;
	float ninv_size;
	//
	int vd_x, vd_y, vd_z;
	float vd_dx, vd_dy, vd_dz;
	//
	int*** hist_1_patch[NumberOfImageChannels];
	int*** hist_2_patch[NumberOfImageChannels];
	float*** vd1_patch_mask;
	float*** vd2_patch_mask;
	const float INVALID_VALUE = -FLT_MAX;
	//
	float* hist_1[NumberOfImageChannels];
	float* hist_2[NumberOfImageChannels];
	float** hist_j[NumberOfImageChannels];
	float hist_j_s[NumberOfImageChannels];
	//
	int i, j, k, l, m, n, d, c;
	//
	double HistMin, HistMax;
	double HistInterval;
	double* pHistogramBuffer;
	int nBins = 20000;
	int total = 0; // total is the total number of plausible matches, used to normalize the histogram
	double Prob;
	double MatchingScoreArr[10];
	double ProbArr[10] = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
	int ii, jj;

	// assume 4 channels for image
	if (d_num > NumberOfImageChannels) {
		return FALSE;
	}

	TRACE2("ComputeDataCost3D_PP_NMI...\n");

	vd_x = vd1[0].m_vd_x;
	vd_y = vd1[0].m_vd_y;
	vd_z = vd1[0].m_vd_z;
	vd_dx = vd1[0].m_vd_dx;
	vd_dy = vd1[0].m_vd_dy;
	vd_dz = vd1[0].m_vd_dz;

	{
		ninv_x = 2 * 4 * ninv_s;
		ninv_y = 2 * 4 * ninv_s;
		ninv_z = 2 * 4 * ninv_s;
		ninv_cx = ninv_x / 2;
		ninv_cy = ninv_y / 2;
		ninv_cz = ninv_z / 2;
		//
		ninv_size = ninv_x * ninv_y * ninv_z;
		ninv = (float***)malloc(ninv_z * sizeof(float**));
		for (k = 0; k < ninv_z; k++) {
			ninv[k] = (float**)malloc(ninv_y * sizeof(float*));
			for (j = 0; j < ninv_y; j++) {
				ninv[k][j] = (float*)malloc(ninv_x * sizeof(float));
			}
		}
		get_ninv(ninv, ninv_x / 4, ninv_y / 4, ninv_z / 4);
	}

	for (c = 0; c < d_num; c++) {
		hist_1[c] = (float*)malloc(NMI_COLOR_NUM * sizeof(float));
		hist_2[c] = (float*)malloc(NMI_COLOR_NUM * sizeof(float));
		hist_j[c] = (float**)malloc(NMI_COLOR_NUM * sizeof(float*));
		for (j = 0; j < NMI_COLOR_NUM; j++) {
			hist_j[c][j] = (float*)malloc(NMI_COLOR_NUM * sizeof(float));
		}
		//
		hist_1_patch[c] = (int***)malloc(ninv_z * sizeof(int**));
		hist_2_patch[c] = (int***)malloc(ninv_z * sizeof(int**));
		for (k = 0; k < ninv_z; k++) {
			hist_1_patch[c][k] = (int**)malloc(ninv_y * sizeof(int*));
			hist_2_patch[c][k] = (int**)malloc(ninv_y * sizeof(int*));
			for (j = 0; j < ninv_y; j++) {
				hist_1_patch[c][k][j] = (int*)malloc(ninv_x * sizeof(int));
				hist_2_patch[c][k][j] = (int*)malloc(ninv_x * sizeof(int));
			}
		}
	}
	vd1_patch_mask = (float***)malloc(ninv_z * sizeof(float**));
	vd2_patch_mask = (float***)malloc(ninv_z * sizeof(float**));
	for (k = 0; k < ninv_z; k++) {
		vd1_patch_mask[k] = (float**)malloc(ninv_y * sizeof(float*));
		vd2_patch_mask[k] = (float**)malloc(ninv_y * sizeof(float*));
		for (j = 0; j < ninv_y; j++) {
			vd1_patch_mask[k][j] = (float*)malloc(ninv_x * sizeof(float));
			vd2_patch_mask[k][j] = (float*)malloc(ninv_x * sizeof(float));
		}
	}

	pHistogramBuffer = new double[nBins];
	memset(pHistogramBuffer, 0, sizeof(double) * nBins);
	HistMin = 10e10;
	HistMax = 0;

	for (n = 0; n < mesh_z; n++) {
		TRACE2("processing %d / %d total z\n", n, mesh_z);
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				float x20, y20, z20;
				float x2, y2, z2;
				float x10, y10, z10;
				float x1, y1, z1;
				float dx, dy, dz;
				float ys;
				int ix1, iy1, iz1;
				int ix2, iy2, iz2;
				float fx, fy, fz, fx1, fy1, fz1;
				//
				float ninv_val;
				float vd1_val, vd2_val;
				int v1, v2;
				double Hj_t[NumberOfImageChannels], H1_t[NumberOfImageChannels], H2_t[NumberOfImageChannels], nmi_t[NumberOfImageChannels];
				float mask1_val, mask2_val;
				float mask_val;
				double mask_sum;
				//
				double corr;
				double corr_c[NumberOfImageChannels];
				BOOL corr_valid;
				float dcv_max, dcv_val;
				//
				for (d = 0; d < num_d; d++) {
					dcv[n][m][l][d] = 0;
				}
				//
				x1 = X1[n][m][l][0] + dX1[n][m][l][0];
				y1 = Y1[n][m][l][0] + dY1[n][m][l][0];
				z1 = Z1[n][m][l][0] + dZ1[n][m][l][0];
				//
				ys = 0;
				for (c = 0; c < d_num; c++) {
					vd1[c].GetAt(x1, y1, z1, &vd1_val);
					ys += vd1_val;
				}
				if (ys == 0) {
					continue;
				}
				//
				//*
				mask1.GetAt(x1, y1, z1, &mask1_val);
				if (mask1_val < 0.5) {
					continue;
				}
				//*/
				//
				x10 = x1 - ninv_cx;
				y10 = y1 - ninv_cy;
				z10 = z1 - ninv_cz;
				//
				for (k = 0; k < ninv_z; k++) {
					for (j = 0; j < ninv_y; j++) {
						for (i = 0; i < ninv_x; i++) {
							x1 = x10 + i;
							y1 = y10 + j;
							z1 = z10 + k;
							//
							if ((x1 <= 0) || (x1 >= vd_x-1) || (y1 <= 0) || (y1 >= vd_y-1) || (z1 <= 0) || (z1 >= vd_z-1)) {
								for (c = 0; c < d_num; c++) {
									hist_1_patch[c][k][j][i] = 0;
								}
								//*
								vd1_patch_mask[k][j][i] = 0;
								/*/
								vd1_patch_mask[k][j][i] = 1;
								//*/
							} else {
								ix1 = (int)x1;
								iy1 = (int)y1;
								iz1 = (int)z1;
								fx = x1 - ix1;
								fy = y1 - iy1;
								fz = z1 - iz1;
								//
								ys = 0;
								if (fx == 0 && fy == 0 && fz == 0) {
									for (c = 0; c < d_num; c++) {
										vd1_val = vd1[c].m_pData[iz1][iy1][ix1][0];
										v1 = ((int)vd1_val) >> NMI_COLOR_SHIFT;
										hist_1_patch[c][k][j][i] = v1;
										ys += vd1_val;
									}
									vd1_patch_mask[k][j][i] = mask1.m_pData[iz1][iy1][ix1][0];
								} else {
									fx1 = 1.0 - fx;
									fy1 = 1.0 - fy;
									fz1 = 1.0 - fz;
									//
									for (c = 0; c < d_num; c++) {
										vd1_val  = fx1*fy1*fz1*vd1[c].m_pData[iz1  ][iy1  ][ix1  ][0];
										vd1_val += fx *fy1*fz1*vd1[c].m_pData[iz1  ][iy1  ][ix1+1][0];
										vd1_val += fx1*fy *fz1*vd1[c].m_pData[iz1  ][iy1+1][ix1  ][0];
										vd1_val += fx1*fy1*fz *vd1[c].m_pData[iz1+1][iy1  ][ix1  ][0];
										vd1_val += fx *fy *fz1*vd1[c].m_pData[iz1  ][iy1+1][ix1+1][0];
										vd1_val += fx *fy1*fz *vd1[c].m_pData[iz1+1][iy1  ][ix1+1][0];
										vd1_val += fx1*fy *fz *vd1[c].m_pData[iz1+1][iy1+1][ix1  ][0];
										vd1_val += fx *fy *fz *vd1[c].m_pData[iz1+1][iy1+1][ix1+1][0];
										v1 = ((int)vd1_val) >> NMI_COLOR_SHIFT;
										hist_1_patch[c][k][j][i] = v1;
										ys += vd1_val;
									}
									//
									mask1_val  = fx1*fy1*fz1*mask1.m_pData[iz1  ][iy1  ][ix1  ][0];
									mask1_val += fx *fy1*fz1*mask1.m_pData[iz1  ][iy1  ][ix1+1][0];
									mask1_val += fx1*fy *fz1*mask1.m_pData[iz1  ][iy1+1][ix1  ][0];
									mask1_val += fx1*fy1*fz *mask1.m_pData[iz1+1][iy1  ][ix1  ][0];
									mask1_val += fx *fy *fz1*mask1.m_pData[iz1  ][iy1+1][ix1+1][0];
									mask1_val += fx *fy1*fz *mask1.m_pData[iz1+1][iy1  ][ix1+1][0];
									mask1_val += fx1*fy *fz *mask1.m_pData[iz1+1][iy1+1][ix1  ][0];
									mask1_val += fx *fy *fz *mask1.m_pData[iz1+1][iy1+1][ix1+1][0];
									vd1_patch_mask[k][j][i] = mask1_val;
								}
								//
								//*
								if (ys == 0) {
									vd1_patch_mask[k][j][i] = 0;
								}
								//*/
							}
						}
					}
				}
				//
				x20 = X2[n][m][l][0] + dX2[n][m][l][0] - ninv_cx;
				y20 = Y2[n][m][l][0] + dY2[n][m][l][0] - ninv_cy;
				z20 = Z2[n][m][l][0] + dZ2[n][m][l][0] - ninv_cz;
				//
				dcv_max = 0;
				for (d = 0; d < num_d; d++) {
					dx = disp_x[d];
					dy = disp_y[d];
					dz = disp_z[d];
					//
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								x2 = x20 + i + dx;
								y2 = y20 + j + dy;
								z2 = z20 + k + dz;
								//
								if ((x2 <= 0) || (x2 >= vd_x-1) || (y2 <= 0) || (y2 >= vd_y-1) || (z2 <= 0) || (z2 >= vd_z-1)) {
									for (c = 0; c < d_num; c++) {
										hist_2_patch[c][k][j][i] = 0;
									}
									//*
									vd2_patch_mask[k][j][i] = 0;
									/*/
									vd2_patch_mask[k][j][i] = 1;
									//*/
								} else {
									ix2 = (int)x2;
									iy2 = (int)y2;
									iz2 = (int)z2;
									fx = x2 - ix2;
									fy = y2 - iy2;
									fz = z2 - iz2;
									//
									fx1 = 1.0 - fx;
									fy1 = 1.0 - fy;
									fz1 = 1.0 - fz;
									//
									ys = 0;
									for (c = 0; c < d_num; c++) {
										vd2_val  = fx1*fy1*fz1*vd2[c].m_pData[iz2  ][iy2  ][ix2  ][0];
										vd2_val += fx *fy1*fz1*vd2[c].m_pData[iz2  ][iy2  ][ix2+1][0];
										vd2_val += fx1*fy *fz1*vd2[c].m_pData[iz2  ][iy2+1][ix2  ][0];
										vd2_val += fx1*fy1*fz *vd2[c].m_pData[iz2+1][iy2  ][ix2  ][0];
										vd2_val += fx *fy *fz1*vd2[c].m_pData[iz2  ][iy2+1][ix2+1][0];
										vd2_val += fx *fy1*fz *vd2[c].m_pData[iz2+1][iy2  ][ix2+1][0];
										vd2_val += fx1*fy *fz *vd2[c].m_pData[iz2+1][iy2+1][ix2  ][0];
										vd2_val += fx *fy *fz *vd2[c].m_pData[iz2+1][iy2+1][ix2+1][0];
										v2 = ((int)vd2_val) >> NMI_COLOR_SHIFT;
										hist_2_patch[c][k][j][i] = v2;
										ys += vd2_val;
									}
									//
									mask2_val  = fx1*fy1*fz1*mask2.m_pData[iz2  ][iy2  ][ix2  ][0];
									mask2_val += fx *fy1*fz1*mask2.m_pData[iz2  ][iy2  ][ix2+1][0];
									mask2_val += fx1*fy *fz1*mask2.m_pData[iz2  ][iy2+1][ix2  ][0];
									mask2_val += fx1*fy1*fz *mask2.m_pData[iz2+1][iy2  ][ix2  ][0];
									mask2_val += fx *fy *fz1*mask2.m_pData[iz2  ][iy2+1][ix2+1][0];
									mask2_val += fx *fy1*fz *mask2.m_pData[iz2+1][iy2  ][ix2+1][0];
									mask2_val += fx1*fy *fz *mask2.m_pData[iz2+1][iy2+1][ix2  ][0];
									mask2_val += fx *fy *fz *mask2.m_pData[iz2+1][iy2+1][ix2+1][0];
									vd2_patch_mask[k][j][i] = mask2_val;
									//
									//*
									if (ys == 0) {
										vd2_patch_mask[k][j][i] = 0;
									}
									//*/
								}
							}
						}
					}
					//
					for (c = 0; c < d_num; c++) {
						hist_j_s[c] = 0;
						for (j = 0; j < NMI_COLOR_NUM; j++) {
							for (i = 0; i < NMI_COLOR_NUM; i++) {
								hist_j[c][j][i] = 0;
							}
							hist_1[c][j] = 0;
							hist_2[c][j] = 0;
						}
					}
					//
					mask_sum = 0;
					for (k = 0; k < ninv_z; k++) {
						for (j = 0; j < ninv_y; j++) {
							for (i = 0; i < ninv_x; i++) {
								double nm;
								ninv_val = ninv[k][j][i];
								mask_val = vd1_patch_mask[k][j][i] * vd2_patch_mask[k][j][i];
								nm = ninv_val * mask_val;
								for (c = 0; c < d_num; c++) {
									v1 = hist_1_patch[c][k][j][i];
									v2 = hist_2_patch[c][k][j][i];
									//
									hist_1[c][v1] += nm;
									hist_2[c][v2] += nm;
									hist_j[c][v2][v1] += nm;
									//
									hist_j_s[c] += nm;
								}
								mask_sum += mask_val;
							}
						}
					}
					//
					if (mask_sum == 0) {
						dcv[n][m][l][d] = INVALID_VALUE;
						continue;
					}
					//
					corr = 0;
					corr_valid = TRUE;
					for (c = 0; c < d_num; c++) {
						if (hist_j_s[c] == 0) {
							//nmi_t[c] = 1;
							corr_valid = FALSE;
							break;
						} else {
							float hv, _hv_s;
							//
							Hj_t[c] = H1_t[c] = H2_t[c] = 0.0;
							_hv_s = 1.0 / hist_j_s[c];
							//
							for (j = 0; j < NMI_COLOR_NUM; j++) {
								for (i = 0; i < NMI_COLOR_NUM; i++) {
									hv = hist_j[c][j][i];
									if (hv != 0) {
										hv *= _hv_s;
										Hj_t[c] += ENT(hv);
									}
								}
								hv = hist_1[c][j];
								if (hv != 0) {
									hv *= _hv_s;
									H1_t[c] += ENT(hv);
								}
								hv = hist_2[c][j];
								if (hv != 0) {
									hv *= _hv_s;
									H2_t[c] += ENT(hv);
								}
							}

							if (Hj_t[c] == 0) {
								//nmi_t[c] = 1;
								corr_valid = FALSE;
								break;
							} else {
								nmi_t[c] = (H1_t[c] + H2_t[c]) / Hj_t[c];
							}
						}

						corr_c[c] = -nmi_t[c];
						corr += corr_c[c];
					}
					if (!corr_valid) {
						dcv[n][m][l][d] = INVALID_VALUE;
					} else {
						//dcv_val = 85 * 256 * (2 + corr / d_num);
						//dcv_val = 128 * 256 * (2 + corr / d_num);
						dcv_val = 256 * 256 * (2 + corr / d_num);
						//dcv_val = 256 * 256 * (2 + (-nmi_t[1]));

						//dcv_val = 85 * 64 * (2 + corr / d_num);
						//dcv_val = 85 * 256 * (2 + (-nmi_t[0]));

						dcv[n][m][l][d] = dcv_val;
						if (dcv_val > dcv_max) {
							dcv_max = dcv_val;
						}
					}
				} // d
				//
				/*
				for (d = 0; d < num_d; d++) {
					if (dcv[n][m][l][d] == INVALID_VALUE) {
						dcv[n][m][l][d] = dcv_max;
					}
				}
				//*/
				//*
				BOOL have_invalid = FALSE;
				for (d = 0; d < num_d; d++) {
					if (dcv[n][m][l][d] == INVALID_VALUE) {
						have_invalid = TRUE;
						break;
					}
				}
				if (have_invalid) {
					for (d = 0; d < num_d; d++) {
						dcv[n][m][l][d] = 0;
					}
				}
				//*/
				//
				for (d = 0; d < num_d; d++) {
					dcv_val = dcv[n][m][l][d];
					if (dcv_val > 0) {
						HistMin = MIN(HistMin, dcv_val);
						HistMax = MAX(HistMax, dcv_val);
						total++;
					}
				}
			} // l
		} // m
	} // n

	// compute the histogram info
	HistInterval = (double)(HistMax - HistMin) / nBins;

	for (n = 0; n < mesh_z; n++) {
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				int val;
				for (d = 0; d < num_d; d++) {
					if (dcv[n][m][l][d] > 0) {
						val = (int)MIN(dcv[n][m][l][d] / HistInterval, nBins-1);
						pHistogramBuffer[val]++;
					}
				}
			}
		}
	}

	// normalize the histogram
	for (ii = 0; ii < nBins; ii++) {
		pHistogramBuffer[ii] /= total;
	}

	// find the matching score
	Prob = 0;
	for (jj = 0; jj < 10; jj++) {
		MatchingScoreArr[jj] = -1;
	}
	for (ii = 0; ii < nBins; ii++) {
		Prob += pHistogramBuffer[ii];
		for (jj = 0; jj < 10; jj++) {
			if (Prob >= ProbArr[jj]) {
				if (MatchingScoreArr[jj] < 0) {
					MatchingScoreArr[jj] = MAX(ii, 1) * HistInterval+HistMin;
				}
			}
		}
	}
	TRACE2("Min: %f\n", HistMin);
	for (jj = 0; jj < 10; jj++) {
		TRACE2("%f, ", MatchingScoreArr[jj]);
	}
	TRACE2("\nMax: %f, total = %d\n", HistMax, total);

	/*
	for (n = 0; n < mesh_z; n++) {
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				for (d = 0; d < num_d; d++) {
					dcv.m_pData[n][m][l][d] = MIN(dcv.m_pData[n][m][l][d], DefaultMatchingScore);
				}
			}
		}
	}
	//*/

	for (k = 0; k < ninv_z; k++) {
		for (j = 0; j < ninv_y; j++) {
			free(ninv[k][j]);
		}
		free(ninv[k]);
	}
	free(ninv);
	//
	for (c = 0; c < d_num; c++) {
		free(hist_1[c]);
		free(hist_2[c]);
		for (j = 0; j < NMI_COLOR_NUM; j++) {
			free(hist_j[c][j]);
		}
		free(hist_j[c]);
		//
		for (k = 0; k < ninv_z; k++) {
			for (j = 0; j < ninv_y; j++) {
				free(hist_1_patch[c][k][j]);
				free(hist_2_patch[c][k][j]);
			}
			free(hist_1_patch[c][k]);
			free(hist_2_patch[c][k]);
		}
		free(hist_1_patch[c]);
		free(hist_2_patch[c]);
	}
	for (k = 0; k < ninv_z; k++) {
		for (j = 0; j < ninv_y; j++) {
			free(vd1_patch_mask[k][j]);
			free(vd2_patch_mask[k][j]);
		}
		free(vd1_patch_mask[k]);
		free(vd2_patch_mask[k]);
	}
	free(vd1_patch_mask);
	free(vd2_patch_mask);
	//
	delete pHistogramBuffer;

	return res;
}

BOOL ComputeDataCost3D_PP(FVolume* vd1, FVolume* vd2, int d_num, FVolume& mask1, FVolume& mask2,
	FVolume* h_x, FVolume* h_y, FVolume* h_z, FVolume& dcv, int lmode, float label_sx, float label_sy, float label_sz, int wsize, int mesh_ex, int mesh_ey, int mesh_ez, int mode,
	float weight = 1.0f) 
{
	int mesh_x = (vd1[0].m_vd_x + mesh_ex-1) / mesh_ex;
	int mesh_y = (vd1[0].m_vd_y + mesh_ey-1) / mesh_ey;
	int mesh_z = (vd1[0].m_vd_z + mesh_ez-1) / mesh_ez;
	int nL = wsize;
    int num_d = 0;
	//
	REALV disp_x[MAX_L3];
	REALV disp_y[MAX_L3];
	REALV disp_z[MAX_L3];
	//
	int K = 2 * nL + 1;
	int K_2 = K * K;
	int i, j, k;
	int dc_skip_back = 1;
	int dc_back_color = 0;

	if (lmode == 0) {
		num_d = K * K * K;
		//
		for (k = -nL; k <= nL; k++) {
			for (j = -nL; j <= nL; j++) {
				for (i = -nL; i <= nL; i++) {
					disp_x[(k+nL)*K_2 + (j+nL)*K + (i+nL)] = label_sx * i;
					disp_y[(k+nL)*K_2 + (j+nL)*K + (i+nL)] = label_sy * j;
					disp_z[(k+nL)*K_2 + (j+nL)*K + (i+nL)] = label_sz * k;
				}
			}
		}
	} else if (lmode == 1) {
		// 26 = 3*3*3 - 1
		int disp_ex[26] = { 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 0, 0, 0, 0, 1, 1, 1,-1, 1,-1,-1,-1 };
		int disp_ey[26] = { 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1 };
		int disp_ez[26] = { 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1 };
		//
		num_d = 26 * nL + 1;
		//
		disp_x[0] = 0;
		disp_y[0] = 0;
		disp_z[0] = 0;
		for (j = 1; j <= nL; j++) {
			for (i = 0; i < 26; i++) {
				disp_x[j + i*nL] = label_sx * j * disp_ex[i];
				disp_y[j + i*nL] = label_sy * j * disp_ey[i];
				disp_z[j + i*nL] = label_sz * j * disp_ez[i];
			}
		}
	} else if (lmode == 2) {
		// 18 = 6 + 4*3
		// (+x, -x, +y, -y, +z, -z, (+x)(+y), (+x)(-y), (-x)(+y), (-x)(-y), (+x)(+z), (+x)(-z), (-x)(+z), (-x)(-z), (+y)(+z), (+y)(-z), (-y)(+z), (-y)(-z) )
		int disp_ex[18] = { 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 0, 0, 0, 0 };
		int disp_ey[18] = { 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1 };
		int disp_ez[18] = { 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1 };
		//
		num_d = 18 * nL + 1;
		//
		disp_x[0] = 0;
		disp_y[0] = 0;
		disp_z[0] = 0;
		for (j = 1; j <= nL; j++) {
			for (i = 0; i < 18; i++) {
				disp_x[j + i*nL] = label_sx * j * disp_ex[i];
				disp_y[j + i*nL] = label_sy * j * disp_ey[i];
				disp_z[j + i*nL] = label_sz * j * disp_ez[i];
			}
		}
	}
	//

	dcv.allocate(mesh_x, mesh_y, mesh_z, num_d);

	{
		RVolume X1, Y1, Z1, dX1, dY1, dZ1, X2, Y2, Z2, dX2, dY2, dZ2;

		X1.allocate(mesh_x, mesh_y, mesh_z);
		Y1.allocate(mesh_x, mesh_y, mesh_z);
		Z1.allocate(mesh_x, mesh_y, mesh_z);
		dX1.allocate(mesh_x, mesh_y, mesh_z);
		dY1.allocate(mesh_x, mesh_y, mesh_z);
		dZ1.allocate(mesh_x, mesh_y, mesh_z);
		X2.allocate(mesh_x, mesh_y, mesh_z);
		Y2.allocate(mesh_x, mesh_y, mesh_z);
		Z2.allocate(mesh_x, mesh_y, mesh_z);
		dX2.allocate(mesh_x, mesh_y, mesh_z);
		dY2.allocate(mesh_x, mesh_y, mesh_z);
		dZ2.allocate(mesh_x, mesh_y, mesh_z);

		#pragma omp parallel for private(i,j,k)
		for (k = 0; k < mesh_z; k++) {
			for (j = 0; j < mesh_y; j++) {
				for (i = 0; i < mesh_x; i++) {
					X1.m_pData[k][j][i][0] = i * mesh_ex;
					Y1.m_pData[k][j][i][0] = j * mesh_ey;
					Z1.m_pData[k][j][i][0] = k * mesh_ez;
					dX1.m_pData[k][j][i][0] = 0;
					dY1.m_pData[k][j][i][0] = 0;
					dZ1.m_pData[k][j][i][0] = 0;
					X2.m_pData[k][j][i][0] = i * mesh_ex;
					Y2.m_pData[k][j][i][0] = j * mesh_ey;
					Z2.m_pData[k][j][i][0] = k * mesh_ez;
					if (h_x != NULL && h_y != NULL && h_z != NULL) {
						dX2.m_pData[k][j][i][0] = h_x->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
						dY2.m_pData[k][j][i][0] = h_y->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
						dZ2.m_pData[k][j][i][0] = h_z->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
					} else {
						dX2.m_pData[k][j][i][0] = 0;
						dY2.m_pData[k][j][i][0] = 0;
						dZ2.m_pData[k][j][i][0] = 0;
					}
				}
			}
		}

		if (mode == 10) {
			// 10 (NCC)
			ComputeDataCost3D_PP_NCC(vd1, vd2, d_num, mask1, mask2,
				X1.m_pData, Y1.m_pData, Z1.m_pData, dX1.m_pData, dY1.m_pData, dZ1.m_pData, X2.m_pData, Y2.m_pData, Z2.m_pData, dX2.m_pData, dY2.m_pData, dZ2.m_pData,
				mesh_x, mesh_y, mesh_z, mesh_ex, mesh_ey, mesh_ez, disp_x, disp_y, disp_z, num_d, dcv.m_pData,
				0, dc_skip_back, (float)dc_back_color);
				//1, dc_skip_back, (float)dc_back_color, 4);
		} else if (mode == 0) {
			// 00 (NMI)
			ComputeDataCost3D_PP_NMI(vd1, vd2, d_num, mask1, mask2,
				X1.m_pData, Y1.m_pData, Z1.m_pData, dX1.m_pData, dY1.m_pData, dZ1.m_pData, X2.m_pData, Y2.m_pData, Z2.m_pData, dX2.m_pData, dY2.m_pData, dZ2.m_pData,
				mesh_x, mesh_y, mesh_z, mesh_ex, mesh_ey, mesh_ez, disp_x, disp_y, disp_z, num_d, dcv.m_pData,
				0, dc_skip_back, (float)dc_back_color);
		} else if (mode == 20) {
			// 20 (CC)
			if (h_x != NULL && h_y != NULL && h_z != NULL) {
				ComputeDataCost3D_PP_CC(vd1, vd2, d_num, mask1, mask2,
					X1.m_pData, Y1.m_pData, Z1.m_pData, dX1.m_pData, dY1.m_pData, dZ1.m_pData, X2.m_pData, Y2.m_pData, Z2.m_pData, dX2.m_pData, dY2.m_pData, dZ2.m_pData,
					mesh_x, mesh_y, mesh_z, mesh_ex, mesh_ey, mesh_ez, disp_x, disp_y, disp_z, num_d, dcv.m_pData,
					dc_skip_back, (float)dc_back_color, 4, weight);
			} else {
				ComputeDataCost3D_PP_CC_Fast(vd1, vd2, d_num, mask1, mask2,
					X1.m_pData, Y1.m_pData, Z1.m_pData,
					mesh_x, mesh_y, mesh_z, mesh_ex, mesh_ey, mesh_ez, disp_x, disp_y, disp_z, num_d, dcv.m_pData,
					dc_skip_back, (float)dc_back_color, 4, weight);
			}
		}
	}

	return TRUE;
}

BOOL AddTumorMatchingCost(FVolume& tu1, FVolume& tu2,
	FVolume* h_x, FVolume* h_y, FVolume* h_z, FVolume& dcv, int lmode, float label_sx, float label_sy, float label_sz, int wsize, int mesh_ex, int mesh_ey, int mesh_ez, 
	float weight = 0.2f)
{
	BOOL res = TRUE;
	//
	int vd_x, vd_y, vd_z;
	float vd_dx, vd_dy, vd_dz;
	//
	int i, j, k, l, m, n, d;
	//
	int mesh_x = (tu1.m_vd_x + mesh_ex-1) / mesh_ex;
	int mesh_y = (tu1.m_vd_y + mesh_ey-1) / mesh_ey;
	int mesh_z = (tu1.m_vd_z + mesh_ez-1) / mesh_ez;
	int nL = wsize;
    int num_d;
	//
	float disp_x[MAX_L3];
	float disp_y[MAX_L3];
	float disp_z[MAX_L3];
	//
	int K = 2 * nL + 1;
	int K_2 = K * K;
	//
	FVolume X1, Y1, Z1, dX1, dY1, dZ1, X2, Y2, Z2, dX2, dY2, dZ2;
	//
	float x1, y1, z1;
	float x20, y20, z20;
	float x2, y2, z2;
	//
	float tu1_val, tu2_val;
	float cost;
	//
#ifdef USE_TE_JS
	float tum_val, js;
#endif

#ifdef USE_TE_L2
	TRACE2("AddTumorMatchingCost (L2)...\n");
#endif
#ifdef USE_TE_JS
	TRACE2("AddTumorMatchingCost (JS)...\n");
#endif

	vd_x = tu1.m_vd_x;
	vd_y = tu1.m_vd_y;
	vd_z = tu1.m_vd_z;
	vd_dx = tu1.m_vd_dx;
	vd_dy = tu1.m_vd_dy;
	vd_dz = tu1.m_vd_dz;

	if (lmode == 0) {
		num_d = K * K * K;
		//
		for (k = -nL; k <= nL; k++) {
			for (j = -nL; j <= nL; j++) {
				for (i = -nL; i <= nL; i++) {
					disp_x[(k+nL)*K_2 + (j+nL)*K + (i+nL)] = label_sx * i;
					disp_y[(k+nL)*K_2 + (j+nL)*K + (i+nL)] = label_sy * j;
					disp_z[(k+nL)*K_2 + (j+nL)*K + (i+nL)] = label_sz * k;
				}
			}
		}
	} else if (lmode == 1) {
		// 26 = 3*3*3 - 1
		int disp_ex[26] = { 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0, 1,-1, 1, 1,-1,-1, 1,-1 };
		int disp_ey[26] = { 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1 };
		int disp_ez[26] = { 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1 };
		//
		num_d = 26 * nL + 1;
		//
		disp_x[0] = 0;
		disp_y[0] = 0;
		disp_z[0] = 0;
		for (j = 1; j <= nL; j++) {
			for (i = 0; i < 26; i++) {
				disp_x[1+26*(j-1)+i] = label_sx * j * disp_ex[i];
				disp_y[1+26*(j-1)+i] = label_sy * j * disp_ey[i];
				disp_z[1+26*(j-1)+i] = label_sz * j * disp_ez[i];
			}
		}
	} else if (lmode == 2) {
		// 18 = 6 + 4*3
		// (+x, -x, +y, -y, +z, -z, (+x)(+y), (+x)(-y), (-x)(+y), (-x)(-y), (+x)(+z), (+x)(-z), (-x)(+z), (-x)(-z), (+y)(+z), (+y)(-z), (-y)(+z), (-y)(-z) )
		int disp_ex[18] = { 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1, 0, 0, 0, 0 };
		int disp_ey[18] = { 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1 };
		int disp_ez[18] = { 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1 };
		//
		num_d = 18 * nL + 1;
		//
		disp_x[0] = 0;
		disp_y[0] = 0;
		disp_z[0] = 0;
		for (j = 1; j <= nL; j++) {
			for (i = 0; i < 18; i++) {
				disp_x[1+18*(j-1)+i] = label_sx * j * disp_ex[i];
				disp_y[1+18*(j-1)+i] = label_sy * j * disp_ey[i];
				disp_z[1+18*(j-1)+i] = label_sz * j * disp_ez[i];
			}
		}
	}
	//

	X1.allocate(mesh_x, mesh_y, mesh_z);
	Y1.allocate(mesh_x, mesh_y, mesh_z);
	Z1.allocate(mesh_x, mesh_y, mesh_z);
	dX1.allocate(mesh_x, mesh_y, mesh_z);
	dY1.allocate(mesh_x, mesh_y, mesh_z);
	dZ1.allocate(mesh_x, mesh_y, mesh_z);
	X2.allocate(mesh_x, mesh_y, mesh_z);
	Y2.allocate(mesh_x, mesh_y, mesh_z);
	Z2.allocate(mesh_x, mesh_y, mesh_z);
	dX2.allocate(mesh_x, mesh_y, mesh_z);
	dY2.allocate(mesh_x, mesh_y, mesh_z);
	dZ2.allocate(mesh_x, mesh_y, mesh_z);

	#pragma omp parallel for private(i,j,k)
	for (k = 0; k < mesh_z; k++) {
		for (j = 0; j < mesh_y; j++) {
			for (i = 0; i < mesh_x; i++) {
				X1.m_pData[k][j][i][0] = i * mesh_ex;
				Y1.m_pData[k][j][i][0] = j * mesh_ey;
				Z1.m_pData[k][j][i][0] = k * mesh_ez;
				dX1.m_pData[k][j][i][0] = 0;
				dY1.m_pData[k][j][i][0] = 0;
				dZ1.m_pData[k][j][i][0] = 0;
				X2.m_pData[k][j][i][0] = i * mesh_ex;
				Y2.m_pData[k][j][i][0] = j * mesh_ey;
				Z2.m_pData[k][j][i][0] = k * mesh_ez;
				if (h_x != NULL && h_y != NULL && h_z != NULL) {
					dX2.m_pData[k][j][i][0] = h_x->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
					dY2.m_pData[k][j][i][0] = h_y->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
					dZ2.m_pData[k][j][i][0] = h_z->m_pData[k * mesh_ez][j * mesh_ey][i * mesh_ex][0];
				} else {
					dX2.m_pData[k][j][i][0] = 0;
					dY2.m_pData[k][j][i][0] = 0;
					dZ2.m_pData[k][j][i][0] = 0;
				}
			}
		}
	}

#ifdef USE_TE_L2
	#pragma omp parallel for private(l,m,n,x1,y1,z1,tu1_val,x20,y20,z20,d,x2,y2,z2,tu2_val,cost)
#endif
#ifdef USE_TE_JS
	#pragma omp parallel for private(l,m,n,x1,y1,z1,tu1_val,x20,y20,z20,d,x2,y2,z2,tu2_val,cost,tum_val,js)
#endif
	for (n = 0; n < mesh_z; n++) {
		TRACE2("processing %d / %d total z\n", n, mesh_z);
		for (m = 0; m < mesh_y; m++) {
			for (l = 0; l < mesh_x; l++) {
				x1 = X1.m_pData[n][m][l][0] + dX1.m_pData[n][m][l][0];
				y1 = Y1.m_pData[n][m][l][0] + dY1.m_pData[n][m][l][0];
				z1 = Z1.m_pData[n][m][l][0] + dZ1.m_pData[n][m][l][0];
				//
				tu1.GetAt(x1, y1, z1, &tu1_val);
				//
				x20 = X2.m_pData[n][m][l][0] + dX2.m_pData[n][m][l][0];
				y20 = Y2.m_pData[n][m][l][0] + dY2.m_pData[n][m][l][0];
				z20 = Z2.m_pData[n][m][l][0] + dZ2.m_pData[n][m][l][0];
				//
				for (d = 0; d < num_d; d++) {
					x2 = x20 + disp_x[d];
					y2 = y20 + disp_y[d];
					z2 = z20 + disp_z[d];
					//
					tu2.GetAt(x2, y2, z2, &tu2_val);
					//
#ifdef USE_TE_L2
					//cost = 256 * 256 * (1.0f - MIN(tu1_val, tu2_val));
					//cost = 256 * 256 * (tu1_val - tu2_val) * (tu1_val - tu2_val);
					cost = 255 * 128 * weight * (tu1_val - tu2_val) * (tu1_val - tu2_val);
#endif
#ifdef USE_TE_JS
					tum_val = (tu1_val + tu2_val) * 0.5;
					js = ENT2(tum_val) - 0.5*(ENT2(tu1_val) + ENT2(tu2_val));
					cost = 255 * 128 * weight * js;
#endif
					//
					dcv.m_pData[n][m][l][d] += cost;
				} // d
			} // l
		} // m
	} // n

	return res;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// dmode: 00 (NMI), 10 (NCC), 20 (CC)
BOOL UpdateDeformationFieldSyM(FVolume* vd1, FVolume* vd2, int d_num, FVolume* tu1, FVolume* tu2, FVolume& ab1, FVolume& ab2, 
	FVolume& h_x, FVolume& h_y, FVolume& h_z, FVolume& h_r_x, FVolume& h_r_y, FVolume& h_r_z,
	char* name, int dmode, int mmode, int m_reg_iter = 2, int m_sub_reg_iter = 3, int m_sub_reg_iter_rep = 1, bool mesh_iter = false,
	bool save_int = true, bool save_pyrd = true, float lambda_D = 1.0, float lambda_P = 0.2,
	float _gamma = -1, float _alpha_O1 = -1, float _d_O1 = -1, float _alpha_O2 = -1, float _d_O2 = -1)
{
	int vd_x, vd_y, vd_z;
	float vd_dx, vd_dy, vd_dz;
	int nlevels;
	//
	float gamma = 0;
	float alpha_O1 = 0;
	float alpha_O2 = 0;
	float d_O1 = 0;
	float d_O2 = 0;
	int nIterations;
	int wsize_a[MAX_LEVELS], wsize;
	int mesh_ex_a[MAX_LEVELS], mesh_ey_a[MAX_LEVELS], mesh_ez_a[MAX_LEVELS], mesh_ex, mesh_ey, mesh_ez;
	float label_sx_a[MAX_LEVELS], label_sy_a[MAX_LEVELS], label_sz_a[MAX_LEVELS], label_sx, label_sy, label_sz;
	float in_scv_w_O1F2 = 0;
	float in_scv_w_O2F2 = 0;
	float in_scv_w_O2F3 = 0;
	//
	int lmode = 0;
	int ffd = 1;
	//
	FVolume xx_cs[MAX_LEVELS], yy_cs[MAX_LEVELS], zz_cs[MAX_LEVELS];
	FVolume xx_ct[MAX_LEVELS], yy_ct[MAX_LEVELS], zz_ct[MAX_LEVELS];
	FVolume vx_cs[MAX_LEVELS], vy_cs[MAX_LEVELS], vz_cs[MAX_LEVELS];
	FVolume vx_ct[MAX_LEVELS], vy_ct[MAX_LEVELS], vz_ct[MAX_LEVELS];
	FVolume vx_cs_r[MAX_LEVELS], vy_cs_r[MAX_LEVELS], vz_cs_r[MAX_LEVELS];
	FVolume vx_ct_r[MAX_LEVELS], vy_ct_r[MAX_LEVELS], vz_ct_r[MAX_LEVELS];
	FVolume wx, wy, wz;
	FVolume wx_r, wy_r, wz_r;
	FVolume tx_cs, ty_cs, tz_cs;
	FVolume tx_ct, ty_ct, tz_ct;
	FVolume hxx_cs, hyy_cs, hzz_cs;
	FVolume hxx_ct, hyy_ct, hzz_ct;
	FVolume hxx_cs_r, hyy_cs_r, hzz_cs_r;
	FVolume hxx_ct_r, hyy_ct_r, hzz_ct_r;
	FVolume hxx, hyy, hzz;
	FVolume hxx_r, hyy_r, hzz_r;
	//
	FVolume pyrd_vd1[MAX_LEVELS][NumberOfImageChannels];
	FVolume pyrd_vd2[MAX_LEVELS][NumberOfImageChannels];
	FVolume pyrd_tu1[MAX_LEVELS], pyrd_tu2[MAX_LEVELS];
	FVolume pyrd_ab1[MAX_LEVELS], pyrd_ab2[MAX_LEVELS];
	int px, py, pz;
	int i, j, k, l, sl, ml;
	int sl_max = 0;
	int ml_max = 0;
	//
	bool use_tumor_energy = false;
	//
	char filename[1024];
	int smode = 1;

	TRACE("Update Deformation...\n");

	vd_x = vd1[0].m_vd_x;
	vd_y = vd1[0].m_vd_y;
	vd_z = vd1[0].m_vd_z;
	vd_dx = vd1[0].m_vd_dx;
	vd_dy = vd1[0].m_vd_dy;
	vd_dz = vd1[0].m_vd_dz;
	//
	{
		int min_size = 32;
		int wmax;
		wmax = max(max(vd_x, vd_y), vd_z);
		nlevels = (int)(log((double)wmax / min_size)/log(2.0) + 1.0);
		if (m_reg_iter > 0) {
			if (m_reg_iter < nlevels) {
				nlevels = m_reg_iter;
			}
		}
	}

	if (tu1 != NULL && tu2 != NULL) {
		use_tumor_energy = true;
	}

	// make image pyramids
	{
		sprintf(filename, "%s_d%02d_pyrd_%d_vd1_%d.nii.gz", name, dmode, 0, 0);
		if (!IsFileExist(filename)) {
			for (i = 0; i < d_num; i++) {
				/*
				FVolume vd1_g;
				FVolume vd2_g;
				vd1_g.allocate(vd_x, vd_y, vd_z);
				vd2_g.allocate(vd_x, vd_y, vd_z);
				vd1[i].GaussianSmoothing(vd1_g, 0.67, 5);
				vd2[i].GaussianSmoothing(vd2_g, 0.67, 5);
				vd1_g.GaussianSmoothing(pyrd_vd1[0][i], 0.67, 5);
				vd2_g.GaussianSmoothing(pyrd_vd2[0][i], 0.67, 5);
				//*/
				pyrd_vd1[0][i].copy(vd1[i]);
				pyrd_vd2[0][i].copy(vd2[i]);
				//vd1[i].GaussianSmoothing(pyrd_vd1[0][i], 0.67, 5);
				//vd2[i].GaussianSmoothing(pyrd_vd2[0][i], 0.67, 5);
				//
				if (save_pyrd) {
					sprintf(filename, "%s_d%02d_pyrd_%d_vd1_%d", name, dmode, 0, i); pyrd_vd1[0][i].save(filename, smode);
					sprintf(filename, "%s_d%02d_pyrd_%d_vd2_%d", name, dmode, 0, i); pyrd_vd2[0][i].save(filename, smode);
				}
				//
				for (k = 1; k < nlevels; k++) {
					pyrd_vd1[k-1][i].GaussianSmoothing(pyrd_vd1[k][i], 0.67, 5);
					pyrd_vd2[k-1][i].GaussianSmoothing(pyrd_vd2[k][i], 0.67, 5);
					pyrd_vd1[k][i].imresize(0.5);
					pyrd_vd2[k][i].imresize(0.5);
					//
					if (save_pyrd) {
						sprintf(filename, "%s_d%02d_pyrd_%d_vd1_%d", name, dmode, k, i); pyrd_vd1[k][i].save(filename, smode);
						sprintf(filename, "%s_d%02d_pyrd_%d_vd2_%d", name, dmode, k, i); pyrd_vd2[k][i].save(filename, smode);
					}
				}
			}
			if (use_tumor_energy) {
				pyrd_tu1[0].copy(*tu1); pyrd_tu2[0].copy(*tu2);
				for (k = 0; k < vd_z; k++) {
					for (j = 0; j < vd_y; j++) {
						for (i = 0; i < vd_x; i++) {
							pyrd_tu1[0].m_pData[k][j][i][0] = 1.0f - tu1->m_pData[k][j][i][0];
							if (pyrd_tu1[0].m_pData[k][j][i][0] > 1.0f) {
								pyrd_tu1[0].m_pData[k][j][i][0] = 1.0f;
							} else if (pyrd_tu1[0].m_pData[k][j][i][0] < 0.0f) {
								pyrd_tu1[0].m_pData[k][j][i][0] = 0.0f;
							}
							pyrd_tu2[0].m_pData[k][j][i][0] = 1.0f - tu2->m_pData[k][j][i][0];
							if (pyrd_tu2[0].m_pData[k][j][i][0] > 1.0f) {
								pyrd_tu2[0].m_pData[k][j][i][0] = 1.0f;
							} else if (pyrd_tu2[0].m_pData[k][j][i][0] < 0.0f) {
								pyrd_tu2[0].m_pData[k][j][i][0] = 0.0f;
							}
						}
					}
				}
				//
				if (save_pyrd) {
					sprintf(filename, "%s_d%02d_pyrd_%d_tu1", name, dmode, 0); pyrd_tu1[0].save(filename, smode);
					sprintf(filename, "%s_d%02d_pyrd_%d_tu2", name, dmode, 0); pyrd_tu2[0].save(filename, smode);
				}
				//
				for (k = 1; k < nlevels; k++) {
					pyrd_tu1[k-1].GaussianSmoothing(pyrd_tu1[k], 0.67, 5); pyrd_tu1[k].imresize(0.5);
					pyrd_tu2[k-1].GaussianSmoothing(pyrd_tu2[k], 0.67, 5); pyrd_tu2[k].imresize(0.5);
					//
					if (save_pyrd) {
						sprintf(filename, "%s_d%02d_pyrd_%d_tu1", name, dmode, k); pyrd_tu1[k].save(filename, smode);
						sprintf(filename, "%s_d%02d_pyrd_%d_tu2", name, dmode, k); pyrd_tu2[k].save(filename, smode);
					}
				}
			}
			{
				pyrd_ab1[0].copy(ab1); pyrd_ab2[0].copy(ab2);
				for (k = 0; k < vd_z; k++) {
					for (j = 0; j < vd_y; j++) {
						for (i = 0; i < vd_x; i++) {
							pyrd_ab1[0].m_pData[k][j][i][0] = 1.0f - ab1.m_pData[k][j][i][0];
							if (pyrd_ab1[0].m_pData[k][j][i][0] > 1.0f) {
								pyrd_ab1[0].m_pData[k][j][i][0] = 1.0f;
							} else if (pyrd_ab1[0].m_pData[k][j][i][0] < 0.0f) {
								pyrd_ab1[0].m_pData[k][j][i][0] = 0.0f;
							}
							pyrd_ab2[0].m_pData[k][j][i][0] = 1.0f - ab2.m_pData[k][j][i][0];
							if (pyrd_ab2[0].m_pData[k][j][i][0] > 1.0f) {
								pyrd_ab2[0].m_pData[k][j][i][0] = 1.0f;
							} else if (pyrd_ab2[0].m_pData[k][j][i][0] < 0.0f) {
								pyrd_ab2[0].m_pData[k][j][i][0] = 0.0f;
							}
						}
					}
				}
				//
				if (save_pyrd) {
					sprintf(filename, "%s_d%02d_pyrd_%d_ab1", name, dmode, 0); pyrd_ab1[0].save(filename, smode);
					sprintf(filename, "%s_d%02d_pyrd_%d_ab2", name, dmode, 0); pyrd_ab2[0].save(filename, smode);
				}
				//
				for (k = 1; k < nlevels; k++) {
					pyrd_ab1[k-1].GaussianSmoothing(pyrd_ab1[k], 0.67, 5); pyrd_ab1[k].imresize(0.5);
 					pyrd_ab2[k-1].GaussianSmoothing(pyrd_ab2[k], 0.67, 5); pyrd_ab2[k].imresize(0.5);
					//
					if (save_pyrd) {
						sprintf(filename, "%s_d%02d_pyrd_%d_ab1", name, dmode, k); pyrd_ab1[k].save(filename, smode);
						sprintf(filename, "%s_d%02d_pyrd_%d_ab2", name, dmode, k); pyrd_ab2[k].save(filename, smode);
					}
				}
			}
			{
				hxx = h_x; hyy = h_y; hzz = h_z;
				//
				//ReverseDeformationField(hxx, hyy, hzz, hxx_r, hyy_r, hzz_r);
				hxx_r = h_r_x; hyy_r = h_r_y; hzz_r = h_r_z;

#ifdef USE_SYMM_REG
				hxx.MultiplyValue(0.5f); hyy.MultiplyValue(0.5f); hzz.MultiplyValue(0.5f);
				hxx_r.MultiplyValue(0.5f); hyy_r.MultiplyValue(0.5f); hzz_r.MultiplyValue(0.5f);

				ReverseDeformationField(hxx, hyy, hzz, xx_cs[0], yy_cs[0], zz_cs[0]);
				ReverseDeformationField(hxx_r, hyy_r, hzz_r, xx_ct[0], yy_ct[0], zz_ct[0]);
#else
				xx_cs[0].allocate(vd_x, vd_y, vd_z); 
				yy_cs[0].allocate(vd_x, vd_y, vd_z);
				zz_cs[0].allocate(vd_x, vd_y, vd_z);
				ReverseDeformationField(hxx_r, hyy_r, hzz_r, xx_ct[0], yy_ct[0], zz_ct[0]);
#endif
				//
				if (save_pyrd) {
					sprintf(filename, "%s_d%02d_pyrd_%d_h_cs.vx3d", name, dmode, 0); xx_cs[0].save(filename, smode);
					sprintf(filename, "%s_d%02d_pyrd_%d_h_cs.vy3d", name, dmode, 0); yy_cs[0].save(filename, smode);
					sprintf(filename, "%s_d%02d_pyrd_%d_h_cs.vz3d", name, dmode, 0); zz_cs[0].save(filename, smode);
					sprintf(filename, "%s_d%02d_pyrd_%d_h_ct.vx3d", name, dmode, 0); xx_ct[0].save(filename, smode);
					sprintf(filename, "%s_d%02d_pyrd_%d_h_ct.vy3d", name, dmode, 0); yy_ct[0].save(filename, smode);
					sprintf(filename, "%s_d%02d_pyrd_%d_h_ct.vz3d", name, dmode, 0); zz_ct[0].save(filename, smode);
				}
				//
				for (k = 1; k < nlevels; k++) {
					xx_cs[k-1].GaussianSmoothing(xx_cs[k], 0.67, 5); xx_cs[k].imresize(0.5); xx_cs[k].MultiplyValue(0.5f);
					yy_cs[k-1].GaussianSmoothing(yy_cs[k], 0.67, 5); yy_cs[k].imresize(0.5); yy_cs[k].MultiplyValue(0.5f);
					zz_cs[k-1].GaussianSmoothing(zz_cs[k], 0.67, 5); zz_cs[k].imresize(0.5); zz_cs[k].MultiplyValue(0.5f);
					xx_ct[k-1].GaussianSmoothing(xx_ct[k], 0.67, 5); xx_ct[k].imresize(0.5); xx_ct[k].MultiplyValue(0.5f);
					yy_ct[k-1].GaussianSmoothing(yy_ct[k], 0.67, 5); yy_ct[k].imresize(0.5); yy_ct[k].MultiplyValue(0.5f);
					zz_ct[k-1].GaussianSmoothing(zz_ct[k], 0.67, 5); zz_ct[k].imresize(0.5); zz_ct[k].MultiplyValue(0.5f);
					//
					if (save_pyrd) {
						sprintf(filename, "%s_d%02d_pyrd_%d_h_cs.vx3d", name, dmode, k); xx_cs[k].save(filename, smode);
						sprintf(filename, "%s_d%02d_pyrd_%d_h_cs.vy3d", name, dmode, k); yy_cs[k].save(filename, smode);
						sprintf(filename, "%s_d%02d_pyrd_%d_h_cs.vz3d", name, dmode, k); zz_cs[k].save(filename, smode);
						sprintf(filename, "%s_d%02d_pyrd_%d_h_ct.vx3d", name, dmode, k); xx_ct[k].save(filename, smode);
						sprintf(filename, "%s_d%02d_pyrd_%d_h_ct.vy3d", name, dmode, k); yy_ct[k].save(filename, smode);
						sprintf(filename, "%s_d%02d_pyrd_%d_h_ct.vz3d", name, dmode, k); zz_ct[k].save(filename, smode);
					}
				}				
			}
		} else {
			for (i = 0; i < d_num; i++) {
				for (k = 0; k < nlevels; k++) {
					sprintf(filename, "%s_d%02d_pyrd_%d_vd1_%d", name, dmode, k, i); pyrd_vd1[k][i].load(filename, smode);
					sprintf(filename, "%s_d%02d_pyrd_%d_vd2_%d", name, dmode, k, i); pyrd_vd2[k][i].load(filename, smode);
				}
			}
			if (use_tumor_energy) {
				for (k = 0; k < nlevels; k++) {
					sprintf(filename, "%s_d%02d_pyrd_%d_tu1", name, dmode, k); pyrd_tu1[k].load(filename, smode);
					sprintf(filename, "%s_d%02d_pyrd_%d_tu2", name, dmode, k); pyrd_tu2[k].load(filename, smode);
				}
			}
			for (k = 0; k < nlevels; k++) {
				sprintf(filename, "%s_d%02d_pyrd_%d_ab1", name, dmode, k); pyrd_ab1[k].load(filename, smode);
				sprintf(filename, "%s_d%02d_pyrd_%d_ab2", name, dmode, k); pyrd_ab2[k].load(filename, smode);
			}
			for (k = 0; k < nlevels; k++) {
				sprintf(filename, "%s_d%02d_pyrd_%d_h_cs.vx3d", name, dmode, k); xx_cs[k].load(filename, smode);
				sprintf(filename, "%s_d%02d_pyrd_%d_h_cs.vy3d", name, dmode, k); yy_cs[k].load(filename, smode);
				sprintf(filename, "%s_d%02d_pyrd_%d_h_cs.vz3d", name, dmode, k); zz_cs[k].load(filename, smode);
				sprintf(filename, "%s_d%02d_pyrd_%d_h_ct.vx3d", name, dmode, k); xx_ct[k].load(filename, smode);
				sprintf(filename, "%s_d%02d_pyrd_%d_h_ct.vy3d", name, dmode, k); yy_ct[k].load(filename, smode);
				sprintf(filename, "%s_d%02d_pyrd_%d_h_ct.vz3d", name, dmode, k); zz_ct[k].load(filename, smode);
			}
		}
	}

	for (l = nlevels-1; l >= 0; l--) {
#if 0
		wsize_a[l] = 4;
#endif
#if 1
		wsize_a[l] = 2;
#endif
	}
	for (l = nlevels-1; l >= 0; l--) {
#if 0
		if (l >= 2) {
			mesh_ex_a[l] = 1;
			mesh_ey_a[l] = 1;
			mesh_ez_a[l] = 1;
		} else if (l == 1) {
			mesh_ex_a[l] = 2;
			mesh_ey_a[l] = 2;
			mesh_ez_a[l] = 2;
		} else if (l == 0) {
			mesh_ex_a[l] = 4;
			mesh_ey_a[l] = 4;
			mesh_ez_a[l] = 4;
		}
#endif
#if 1
		mesh_ex_a[l] = 4;
		mesh_ey_a[l] = 4;
		mesh_ez_a[l] = 4;
#endif
	}
	for (l = nlevels-1; l >= 0; l--) {
#if 0
		label_sx_a[l] = 0.1;
		label_sy_a[l] = 0.1;
		label_sz_a[l] = 0.1;
#endif
#if 0
		label_sx_a[l] = 0.4;
		label_sy_a[l] = 0.4;
		label_sz_a[l] = 0.4;
#endif
#if 0
		label_sx_a[l] = 1.0;
		label_sy_a[l] = 1.0;
		label_sz_a[l] = 1.0;
#endif
#if 1
		label_sx_a[l] = mesh_ex_a[l] * 0.4 / wsize_a[l];
		label_sy_a[l] = mesh_ey_a[l] * 0.4 / wsize_a[l];
		label_sz_a[l] = mesh_ez_a[l] * 0.4 / wsize_a[l];
#endif
	}

	if (_gamma == -1) {
		if (mmode == 0) {
			gamma			= 0 * 255;
			alpha_O1		= 0.1 * 255;
			d_O1			= 10 * 255;
			alpha_O2		= 0;
			d_O2			= 0;
			//
			nIterations		= 200;
		} else if (mmode == 11) {
			gamma			= 0 * 255;
			alpha_O1		= 0;
			d_O1			= 0;
			alpha_O2		= 0.1 * 255;
			d_O2			= 10 * 255;
			//
			nIterations		= 200;
		} else if (mmode == 22) {
			gamma			= 0 * 255;
			alpha_O1		= 0.1 * 255;
			d_O1			= 10 * 255;
			alpha_O2		= 0.1 * 255;
			d_O2			= 10 * 255;
			//
			nIterations		= 200;
		}
	} else {
		gamma			= _gamma * 255;
		alpha_O1		= _alpha_O1 * 255;
		d_O1			= _d_O1 * 255;
		alpha_O2		= _alpha_O2 * 255;
		d_O2			= _d_O2 * 255;
		//
		nIterations		= 200;
	}
	if (mmode == 0) {
		in_scv_w_O1F2 = 1.0;
		in_scv_w_O2F2 = -2;
		in_scv_w_O2F3 = -2;
	} else if (mmode == 11) {
		in_scv_w_O1F2 = -2;
		in_scv_w_O2F2 = -2;
		in_scv_w_O2F3 = 1.0;
	} else if (mmode == 22) {
		in_scv_w_O1F2 = 1.0;
		in_scv_w_O2F2 = -2;
		in_scv_w_O2F3 = 1.0;
	}

	for (l = nlevels-1; l >= 0; l--) {
		TRACE("level = %d\n", l);

		FVolume pyrd_vd1_warp[NumberOfImageChannels];
		FVolume pyrd_vd2_warp[NumberOfImageChannels];
		FVolume pyrd_tu1_warp;
		FVolume pyrd_tu2_warp;
		FVolume pyrd_ab1_warp;
		FVolume pyrd_ab2_warp;
		//
		// normalize image
		for (i = 0; i < d_num; i++) {
			pyrd_vd1[l][i].NormalizeImage();
			pyrd_vd2[l][i].NormalizeImage();
		}
		//
		px = pyrd_vd1[l][0].m_vd_x;
		py = pyrd_vd1[l][0].m_vd_y;
		pz = pyrd_vd1[l][0].m_vd_z;
		// allocate
		vx_cs[l].allocate(px, py, pz); vy_cs[l].allocate(px, py, pz); vz_cs[l].allocate(px, py, pz);
		vx_ct[l].allocate(px, py, pz); vy_ct[l].allocate(px, py, pz); vz_ct[l].allocate(px, py, pz);
		vx_cs_r[l].allocate(px, py, pz); vy_cs_r[l].allocate(px, py, pz); vz_cs_r[l].allocate(px, py, pz);
		vx_ct_r[l].allocate(px, py, pz); vy_ct_r[l].allocate(px, py, pz); vz_ct_r[l].allocate(px, py, pz);
		wx.allocate(px, py, pz); wy.allocate(px, py, pz); wz.allocate(px, py, pz);
		wx_r.allocate(px, py, pz); wy_r.allocate(px, py, pz); wz_r.allocate(px, py, pz);
		tx_cs.allocate(px, py, pz); ty_cs.allocate(px, py, pz); tz_cs.allocate(px, py, pz);
		tx_ct.allocate(px, py, pz); ty_ct.allocate(px, py, pz); tz_ct.allocate(px, py, pz);
		//
		if (l == nlevels-1) {
		} else {
			vx_cs[l+1].imresize(vx_cs[l], px, py, pz, 1); vy_cs[l+1].imresize(vy_cs[l], px, py, pz, 1); vz_cs[l+1].imresize(vz_cs[l], px, py, pz, 1);
			vx_ct[l+1].imresize(vx_ct[l], px, py, pz, 1); vy_ct[l+1].imresize(vy_ct[l], px, py, pz, 1); vz_ct[l+1].imresize(vz_ct[l], px, py, pz, 1);
			vx_cs_r[l+1].imresize(vx_cs_r[l], px, py, pz, 1); vy_cs_r[l+1].imresize(vy_cs_r[l], px, py, pz, 1); vz_cs_r[l+1].imresize(vz_cs_r[l], px, py, pz, 1);
			vx_ct_r[l+1].imresize(vx_ct_r[l], px, py, pz, 1); vy_ct_r[l+1].imresize(vy_ct_r[l], px, py, pz, 1); vz_ct_r[l+1].imresize(vz_ct_r[l], px, py, pz, 1);
			for (k = 0; k < pz; k++) {
				for (j = 0; j < py; j++) {
					for (i = 0; i < px; i++) {
						vx_cs[l].m_pData[k][j][i][0] *= 2.0;
						vy_cs[l].m_pData[k][j][i][0] *= 2.0;
						vz_cs[l].m_pData[k][j][i][0] *= 2.0;
						vx_ct[l].m_pData[k][j][i][0] *= 2.0;
						vy_ct[l].m_pData[k][j][i][0] *= 2.0;
						vz_ct[l].m_pData[k][j][i][0] *= 2.0;
						vx_cs_r[l].m_pData[k][j][i][0] *= 2.0;
						vy_cs_r[l].m_pData[k][j][i][0] *= 2.0;
						vz_cs_r[l].m_pData[k][j][i][0] *= 2.0;
						vx_ct_r[l].m_pData[k][j][i][0] *= 2.0;
						vy_ct_r[l].m_pData[k][j][i][0] *= 2.0;
						vz_ct_r[l].m_pData[k][j][i][0] *= 2.0;
					}
				}
			}
			vx_cs[l+1].clear(); vy_cs[l+1].clear(); vz_cs[l+1].clear();
			vx_ct[l+1].clear(); vy_ct[l+1].clear(); vz_ct[l+1].clear();
			vx_cs_r[l+1].clear(); vy_cs_r[l+1].clear(); vz_cs_r[l+1].clear();
			vx_ct_r[l+1].clear(); vy_ct_r[l+1].clear(); vz_ct_r[l+1].clear();
		}

		if (mesh_iter) {
			if (l >= 2) {
				ml_max = 2;
			} else if (l == 1) {
				ml_max = 2;
			} else if (l == 0) {
				ml_max = 1;
			}
		} else {
			ml_max = 0;
		}

		if (l >= 2) {
			sl_max = m_sub_reg_iter*m_sub_reg_iter_rep;
		} else if (l == 1) {
			sl_max = m_sub_reg_iter*m_sub_reg_iter_rep;
		} else if (l == 0) {
			sl_max = m_sub_reg_iter*m_sub_reg_iter_rep - 1;
		}

#ifdef USE_DISCRETE_OPTIMIZATION
		TRACE("Update Deformation (Discrete)...\n");
		//
		sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs.vx3d", name, l, ml_max, sl_max, dmode, mmode);
		if (!vx_cs[l].load(filename, smode)) 
		{
			for (ml = 0; ml <= ml_max; ml++) 
			{
				mesh_ex = (int)(mesh_ex_a[l] / pow(2.0, ml));
				mesh_ey = (int)(mesh_ey_a[l] / pow(2.0, ml));
				mesh_ez = (int)(mesh_ez_a[l] / pow(2.0, ml));
				//wsize = wsize_a[l] / pow(2.0, ml);
				wsize = wsize_a[l];
				if (mesh_ex < 1 || mesh_ey < 1 || mesh_ez < 1 || wsize < 1) {
					continue;
				}

				for (sl = 0; sl <= sl_max; sl++)
				{
					FVolume dcv;
					//
					if (mesh_iter && ml > 0) {
						label_sx = (mesh_ex * 0.4 / wsize) / pow(2.0, sl/m_sub_reg_iter_rep);
						label_sy = (mesh_ey * 0.4 / wsize) / pow(2.0, sl/m_sub_reg_iter_rep);
						label_sz = (mesh_ez * 0.4 / wsize) / pow(2.0, sl/m_sub_reg_iter_rep);
					} else {
						label_sx = label_sx_a[l] / pow(2.0, sl/m_sub_reg_iter_rep);
						label_sy = label_sy_a[l] / pow(2.0, sl/m_sub_reg_iter_rep);
						label_sz = label_sz_a[l] / pow(2.0, sl/m_sub_reg_iter_rep);
					}

					TRACE2("\nl = %d, ml = %d, sl = %d\n", l, ml, sl);
					TRACE2("mesh_ex = %d, mesh_ey = %d, mesh_ez = %d\n", mesh_ex, mesh_ey, mesh_ez);
					TRACE2("wsize = %d\n", wsize);
					TRACE2("label_sx = %f, label_sy = %f, label_sz = %f\n\n", label_sx, label_sy, label_sz);

					ComposeFields(vx_cs[l], vy_cs[l], vz_cs[l], xx_cs[l], yy_cs[l], zz_cs[l], hxx_cs, hyy_cs, hzz_cs, 1.0);
					ComposeFields(vx_ct[l], vy_ct[l], vz_ct[l], xx_ct[l], yy_ct[l], zz_ct[l], hxx_ct, hyy_ct, hzz_ct, 1.0);

					{
						// warp pyrd_vd1, pyrd_vd2
						for (i = 0; i < d_num; i++) {
							WarpImage(pyrd_vd1[l][i], pyrd_vd1_warp[i], hxx_cs, hyy_cs, hzz_cs);
							WarpImage(pyrd_vd2[l][i], pyrd_vd2_warp[i], hxx_ct, hyy_ct, hzz_ct);
						}
						if (use_tumor_energy) {
							WarpImage(pyrd_tu1[l], pyrd_tu1_warp, hxx_cs, hyy_cs, hzz_cs, 1.0f);
							WarpImage(pyrd_tu2[l], pyrd_tu2_warp, hxx_ct, hyy_ct, hzz_ct, 1.0f);
						}
						WarpImage(pyrd_ab1[l], pyrd_ab1_warp, hxx_cs, hyy_cs, hzz_cs, 1.0f);
						WarpImage(pyrd_ab2[l], pyrd_ab2_warp, hxx_ct, hyy_ct, hzz_ct, 1.0f);

						//
						{
							double energy;
							double energy_val[10];
							energy = 0;
							for (i = 0; i < d_num; i++) {
								energy_val[i] = GetEnergy_CC(pyrd_vd1_warp[i], pyrd_vd2_warp[i], &pyrd_ab1_warp, &pyrd_ab2_warp, 0) ;
								energy += energy_val[i];
							}
							energy /= d_num;
							TRACE2("energy = %f\n\n", energy);
						}
						//

#ifdef USE_SYMM_REG
						sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_cs_r.dcv", name, l, ml, sl, dmode, mmode);
						if (!dcv.load(filename, smode)) {
							ComputeDataCost3D_PP(pyrd_vd1_warp, pyrd_vd2_warp, d_num, pyrd_ab1_warp, pyrd_ab2_warp,
								NULL, NULL, NULL,
								dcv, lmode, label_sx, label_sy, label_sz, wsize, mesh_ex, mesh_ey, mesh_ez, dmode, lambda_D);
							//
							if (use_tumor_energy) {
								AddTumorMatchingCost(pyrd_tu1_warp, pyrd_tu2_warp,
									NULL, NULL, NULL,
									dcv, lmode, label_sx, label_sy, label_sz, wsize, mesh_ex, mesh_ey, mesh_ez, lambda_D*lambda_P);
							}
							//
							if (save_int) {
							//if (save_int || sl == sl_max) {
							//{
								sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_cs_r.dcv", name, l, ml, sl, dmode, mmode);
								dcv.save(filename, smode);
							}
						}

						{
							ComputeFlow3D_TRWS_Decomposed(dcv, wsize, label_sx, label_sy, label_sz, lmode, alpha_O1, d_O1, alpha_O2, d_O2, gamma, nIterations,
								NULL, NULL, NULL,
								wx, wy, wz, px, py, pz, mesh_ex, mesh_ey, mesh_ez,
								in_scv_w_O1F2, in_scv_w_O2F2, in_scv_w_O2F3, NULL, NULL, NULL, ffd);

							if (save_int) {
								sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_cs_r.vx3d", name, l, ml, sl, dmode, mmode); wx.save(filename, smode);
								sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_cs_r.vy3d", name, l, ml, sl, dmode, mmode); wy.save(filename, smode);
								sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_cs_r.vz3d", name, l, ml, sl, dmode, mmode); wz.save(filename, smode);
							}
						}

						dcv.clear();
#else
						wx.setValue(0.0f); wy.setValue(0.0f); wz.setValue(0.0f);
#endif

						sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_ct_r.dcv", name, l, ml, sl, dmode, mmode);
						if (!dcv.load(filename, smode)) {
							ComputeDataCost3D_PP(pyrd_vd2_warp, pyrd_vd1_warp, d_num, pyrd_ab2_warp, pyrd_ab1_warp,
								NULL, NULL, NULL,
								dcv, lmode, label_sx, label_sy, label_sz, wsize, mesh_ex, mesh_ey, mesh_ez, dmode, lambda_D);
							//
							if (use_tumor_energy) {
								AddTumorMatchingCost(pyrd_tu2_warp, pyrd_tu1_warp,
									NULL, NULL, NULL,
									dcv, lmode, label_sx, label_sy, label_sz, wsize, mesh_ex, mesh_ey, mesh_ez, lambda_D*lambda_P);
							}
							//
							if (save_int) {
							//if (save_int || sl == sl_max) {
							//{
								sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_ct_r.dcv", name, l, ml, sl, dmode, mmode);
								dcv.save(filename, smode);
							}
						}

						{
							ComputeFlow3D_TRWS_Decomposed(dcv, wsize, label_sx, label_sy, label_sz, lmode, alpha_O1, d_O1, alpha_O2, d_O2, gamma, nIterations,
								NULL, NULL, NULL,
								wx_r, wy_r, wz_r, px, py, pz, mesh_ex, mesh_ey, mesh_ez,
								in_scv_w_O1F2, in_scv_w_O2F2, in_scv_w_O2F3, NULL, NULL, NULL, ffd);

							if (save_int) {
								sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_ct_r.vx3d", name, l, ml, sl, dmode, mmode); wx_r.save(filename, smode);
								sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_ct_r.vy3d", name, l, ml, sl, dmode, mmode); wy_r.save(filename, smode);
								sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_ct_r.vz3d", name, l, ml, sl, dmode, mmode); wz_r.save(filename, smode);
							}
						}

						dcv.clear();

						for (i = 0; i < d_num; i++) {
							pyrd_vd1_warp[i].clear();
							pyrd_vd2_warp[i].clear();
						}
						if (use_tumor_energy) {
							pyrd_tu1_warp.clear();
							pyrd_tu2_warp.clear();
						}
						pyrd_ab1_warp.clear();
						pyrd_ab2_warp.clear();
					}

					{
#ifdef USE_SYMM_REG
						ComposeFields(vx_cs_r[l], vy_cs_r[l], vz_cs_r[l], wx, wy, wz, tx_cs, ty_cs, tz_cs, 0.5);
						ComposeFields(vx_ct_r[l], vy_ct_r[l], vz_ct_r[l], wx_r, wy_r, wz_r, tx_ct, ty_ct, tz_ct, 0.5);
#else
						ComposeFields(vx_cs_r[l], vy_cs_r[l], vz_cs_r[l], wx, wy, wz, tx_cs, ty_cs, tz_cs, 1.0);
						ComposeFields(vx_ct_r[l], vy_ct_r[l], vz_ct_r[l], wx_r, wy_r, wz_r, tx_ct, ty_ct, tz_ct, 1.0);
#endif

						vx_cs_r[l] = tx_cs; vy_cs_r[l] = ty_cs; vz_cs_r[l] = tz_cs;
						vx_ct_r[l] = tx_ct; vy_ct_r[l] = ty_ct; vz_ct_r[l] = tz_ct;

						ReverseField(vx_cs_r[l], vy_cs_r[l], vz_cs_r[l], vx_cs[l], vy_cs[l], vz_cs[l]);
						ReverseField(vx_ct_r[l], vy_ct_r[l], vz_ct_r[l], vx_ct[l], vy_ct[l], vz_ct[l]);
						ReverseField(vx_cs[l], vy_cs[l], vz_cs[l], vx_cs_r[l], vy_cs_r[l], vz_cs_r[l]);
						ReverseField(vx_ct[l], vy_ct[l], vz_ct[l], vx_ct_r[l], vy_ct_r[l], vz_ct_r[l]);

						if (save_int) {
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs.vx3d", name, l, ml, sl, dmode, mmode); vx_cs[l].save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs.vy3d", name, l, ml, sl, dmode, mmode); vy_cs[l].save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs.vz3d", name, l, ml, sl, dmode, mmode); vz_cs[l].save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct.vx3d", name, l, ml, sl, dmode, mmode); vx_ct[l].save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct.vy3d", name, l, ml, sl, dmode, mmode); vy_ct[l].save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct.vz3d", name, l, ml, sl, dmode, mmode); vz_ct[l].save(filename, smode);
						}
						if (save_int && sl == sl_max) {
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs_r.vx3d", name, l, ml, sl, dmode, mmode); vx_cs_r[l].save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs_r.vy3d", name, l, ml, sl, dmode, mmode); vy_cs_r[l].save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs_r.vz3d", name, l, ml, sl, dmode, mmode); vz_cs_r[l].save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct_r.vx3d", name, l, ml, sl, dmode, mmode); vx_ct_r[l].save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct_r.vy3d", name, l, ml, sl, dmode, mmode); vy_ct_r[l].save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct_r.vz3d", name, l, ml, sl, dmode, mmode); vz_ct_r[l].save(filename, smode);
						}
					}
									
					//if (save_int && sl == sl_max) {
					if (save_int) {
					//{
						ComposeFields(vx_cs[l], vy_cs[l], vz_cs[l], xx_cs[l], yy_cs[l], zz_cs[l], hxx_cs, hyy_cs, hzz_cs, 1.0);
						ComposeFields(vx_ct[l], vy_ct[l], vz_ct[l], xx_ct[l], yy_ct[l], zz_ct[l], hxx_ct, hyy_ct, hzz_ct, 1.0);

						ReverseDeformationField(hxx_cs, hyy_cs, hzz_cs, hxx_cs_r, hyy_cs_r, hzz_cs_r);
						ReverseDeformationField(hxx_ct, hyy_ct, hzz_ct, hxx_ct_r, hyy_ct_r, hzz_ct_r);

						// reverse cs and concatenate with ct
						ComposeFields(hxx_cs_r, hyy_cs_r, hzz_cs_r, hxx_ct, hyy_ct, hzz_ct, hxx, hyy, hzz, 1.0);
						// reverse ct and concatenate with cs
						ComposeFields(hxx_ct_r, hyy_ct_r, hzz_ct_r, hxx_cs, hyy_cs, hzz_cs, hxx_r, hyy_r, hzz_r, 1.0);

						{
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h.vx3d", name, l, ml, sl, dmode, mmode); hxx.save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h.vy3d", name, l, ml, sl, dmode, mmode); hyy.save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h.vz3d", name, l, ml, sl, dmode, mmode); hzz.save(filename, smode);
						}
						if (sl == sl_max) {
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_r.vx3d", name, l, ml, sl, dmode, mmode); hxx_r.save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_r.vy3d", name, l, ml, sl, dmode, mmode); hyy_r.save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_r.vz3d", name, l, ml, sl, dmode, mmode); hzz_r.save(filename, smode);
							//
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_cs.vx3d", name, l, ml, sl, dmode, mmode); hxx_cs.save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_cs.vy3d", name, l, ml, sl, dmode, mmode); hyy_cs.save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_cs.vz3d", name, l, ml, sl, dmode, mmode); hzz_cs.save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_ct.vx3d", name, l, ml, sl, dmode, mmode); hxx_ct.save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_ct.vy3d", name, l, ml, sl, dmode, mmode); hyy_ct.save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_ct.vz3d", name, l, ml, sl, dmode, mmode); hzz_ct.save(filename, smode);
						}
						//
						// warp images
						{
							FVolume vd1t[NumberOfImageChannels];
							FVolume vd2t[NumberOfImageChannels];
							for (i = 0; i < d_num; i++) {
								vd1t[i].allocate(pyrd_vd1[l][i].m_vd_x, pyrd_vd1[l][i].m_vd_y, pyrd_vd1[l][i].m_vd_z);
								vd2t[i].allocate(pyrd_vd2[l][i].m_vd_x, pyrd_vd2[l][i].m_vd_y, pyrd_vd2[l][i].m_vd_z);

								GenerateBackwardWarpVolume(vd1t[i], pyrd_vd1[l][i], hxx_cs, hyy_cs, hzz_cs, 0.0f, false);
								GenerateBackwardWarpVolume(vd2t[i], pyrd_vd2[l][i], hxx_ct, hyy_ct, hzz_ct, 0.0f, false);

								sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_vd1tc_%d", name, l, ml, sl, dmode, mmode, i); vd1t[i].save(filename, smode);
								sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_vd2tc_%d", name, l, ml, sl, dmode, mmode, i); vd2t[i].save(filename, smode);

								GenerateBackwardWarpVolume(vd1t[i], pyrd_vd1[l][i], hxx_r, hyy_r, hzz_r, 0.0f, false);
								GenerateBackwardWarpVolume(vd2t[i], pyrd_vd2[l][i], hxx, hyy, hzz, 0.0f, false);

								sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_vd1t_%d", name, l, ml, sl, dmode, mmode, i); vd1t[i].save(filename, smode);
								sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_vd2t_%d", name, l, ml, sl, dmode, mmode, i); vd2t[i].save(filename, smode);

								vd1t[i].clear();
								vd2t[i].clear();
							}
						}
					}
				} // sl
			} // ml
		} else {
			sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs.vy3d", name, l, ml_max, sl_max, dmode, mmode); vy_cs[l].load(filename, smode);
			sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs.vz3d", name, l, ml_max, sl_max, dmode, mmode); vz_cs[l].load(filename, smode);
			sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct.vx3d", name, l, ml_max, sl_max, dmode, mmode); vx_ct[l].load(filename, smode);
			sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct.vy3d", name, l, ml_max, sl_max, dmode, mmode); vy_ct[l].load(filename, smode);
			sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct.vz3d", name, l, ml_max, sl_max, dmode, mmode); vz_ct[l].load(filename, smode);
			sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs_r.vx3d", name, l, ml_max, sl_max, dmode, mmode); vx_cs_r[l].load(filename, smode);
			sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs_r.vy3d", name, l, ml_max, sl_max, dmode, mmode); vy_cs_r[l].load(filename, smode);
			sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs_r.vz3d", name, l, ml_max, sl_max, dmode, mmode); vz_cs_r[l].load(filename, smode);
			sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct_r.vx3d", name, l, ml_max, sl_max, dmode, mmode); vx_ct_r[l].load(filename, smode);
			sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct_r.vy3d", name, l, ml_max, sl_max, dmode, mmode); vy_ct_r[l].load(filename, smode);
			sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct_r.vz3d", name, l, ml_max, sl_max, dmode, mmode); vz_ct_r[l].load(filename, smode);
		}
		//
		TRACE("Update Deformation (Discrete)...done\n");
#endif

#ifdef USE_CONTINUOUS_OPTIMIZATION
		TRACE("Update Deformation (Continuous)...\n");

		{
			int it;
			int iter = 0;
#ifdef USE_SYMM_REG
			float grad_step = 0.5;
#else
			float grad_step = 1.0;
#endif
			double energy_list[100];

			if (l >= 2) {
				iter = CONT_OPT_ITER_LEVEL_N;
			} else if (l == 1) {
				iter = CONT_OPT_ITER_LEVEL_N;
			} else if (l == 0) {
				iter = CONT_OPT_ITER_LEVEL_0;
			}

			for (it = 0; it < iter; it++) {
				ComposeFields(vx_cs[l], vy_cs[l], vz_cs[l], xx_cs[l], yy_cs[l], zz_cs[l], hxx_cs, hyy_cs, hzz_cs, 1.0);
				ComposeFields(vx_ct[l], vy_ct[l], vz_ct[l], xx_ct[l], yy_ct[l], zz_ct[l], hxx_ct, hyy_ct, hzz_ct, 1.0);

				// warp pyrd_vd1, pyrd_vd2
				for (i = 0; i < d_num; i++) {
					WarpImage(pyrd_vd1[l][i], pyrd_vd1_warp[i], hxx_cs, hyy_cs, hzz_cs);
					WarpImage(pyrd_vd2[l][i], pyrd_vd2_warp[i], hxx_ct, hyy_ct, hzz_ct);
				}
				if (use_tumor_energy) {
					WarpImage(pyrd_tu1[l], pyrd_tu1_warp, hxx_cs, hyy_cs, hzz_cs, 1.0f);
					WarpImage(pyrd_tu2[l], pyrd_tu2_warp, hxx_ct, hyy_ct, hzz_ct, 1.0f);
				}
				WarpImage(pyrd_ab1[l], pyrd_ab1_warp, hxx_cs, hyy_cs, hzz_cs, 1.0f);
				WarpImage(pyrd_ab2[l], pyrd_ab2_warp, hxx_ct, hyy_ct, hzz_ct, 1.0f);

				//*
				{
					FVolume wxx, wyy, wzz;
					FVolume wxx_r, wyy_r, wzz_r;
					double energy_val[10];
					//
					wxx.allocate(px, py, pz); wyy.allocate(px, py, pz); wzz.allocate(px, py, pz);					
					wxx_r.allocate(px, py, pz); wyy_r.allocate(px, py, pz); wzz_r.allocate(px, py, pz);					
					//
					energy_list[it] = 0;
					for (i = 0; i < d_num; i++) {
						energy_val[i] = GetVelocity_CC(pyrd_vd1_warp[i], pyrd_vd2_warp[i], &pyrd_ab1_warp, &pyrd_ab2_warp, &wxx, &wyy, &wzz, &wxx_r, &wyy_r, &wzz_r, 0, 0, lambda_D);
						//
						if (i == 0) {
#ifdef USE_SYMM_REG
							wx = wxx; wy = wyy; wz = wzz;
#endif
							wx_r = wxx_r; wy_r = wyy_r; wz_r = wzz_r;
						} else {
#ifdef USE_SYMM_REG
							wx += wxx; wy += wyy; wz += wzz;
#endif
							wx_r += wxx_r; wy_r += wyy_r; wz_r += wzz_r;
						}
						//
						energy_list[it] += energy_val[i];
					}
					energy_list[it] /= d_num;
					TRACE2("it = %d, energy = %f\n", it, energy_list[it]);
					//
#ifdef USE_SYMM_REG
					wx.MultiplyValue(1.0f / d_num); 
					wy.MultiplyValue(1.0f / d_num); 
					wz.MultiplyValue(1.0f / d_num);
#else
					wx.setValue(0.0f); wy.setValue(0.0f); wz.setValue(0.0f);
#endif
					//
					wx_r.MultiplyValue(1.0f / d_num); 
					wy_r.MultiplyValue(1.0f / d_num); 
					wz_r.MultiplyValue(1.0f / d_num);
					//
					if (use_tumor_energy) {
						GetVelocity_Tumor(pyrd_tu1_warp, pyrd_tu2_warp, &wxx, &wyy, &wzz, &wxx_r, &wyy_r, &wzz_r, lambda_D*lambda_P);
						//
#ifdef USE_SYMM_REG
						wx += wxx; wy += wyy; wz += wzz;
#endif
						wx_r += wxx_r; wy_r += wyy_r; wz_r += wzz_r;
					}
					//
					wxx.clear(); wyy.clear(); wzz.clear();
					wxx_r.clear(); wyy_r.clear(); wzz_r.clear();
				}
				/*/
				energy_list[it] = GetVelocity_CC(pyrd_vd1_warp[0], pyrd_vd2_warp[0], &pyrd_ab1_warp, &pyrd_ab2_warp, &wx, &wy, &wz, &wx_r, &wy_r, &wz_r, 0);
				//energy_list[it] = GetVelocity_CC(pyrd_vd1_warp[1], pyrd_vd2_warp[1], &pyrd_ab1_warp, &pyrd_ab2_warp, &wx, &wy, &wz, &wx_r, &wy_r, &wz_r, 0);
				TRACE2("it = %d, energy = %f\n", it, energy_list[it]);
				//*/

				for (i = 0; i < d_num; i++) {					
					pyrd_vd1_warp[i].clear();
					pyrd_vd2_warp[i].clear();
				}
				if (use_tumor_energy) {
					pyrd_tu1_warp.clear();
					pyrd_tu2_warp.clear();
				}
				pyrd_ab1_warp.clear();
				pyrd_ab2_warp.clear();

				/*
				sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_cs_r.vx3d", name, l, ml_max+1, it, dmode, mmode); wx.save(filename, smode);
				sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_cs_r.vy3d", name, l, ml_max+1, it, dmode, mmode); wy.save(filename, smode);
				sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_cs_r.vz3d", name, l, ml_max+1, it, dmode, mmode); wz.save(filename, smode);
				sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_ct_r.vx3d", name, l, ml_max+1, it, dmode, mmode); wx_r.save(filename, smode);
				sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_ct_r.vy3d", name, l, ml_max+1, it, dmode, mmode); wy_r.save(filename, smode);
				sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_w_ct_r.vz3d", name, l, ml_max+1, it, dmode, mmode); wz_r.save(filename, smode);
				//*/

				ComposeFields(vx_cs_r[l], vy_cs_r[l], vz_cs_r[l], wx, wy, wz, tx_cs, ty_cs, tz_cs, grad_step);
				ComposeFields(vx_ct_r[l], vy_ct_r[l], vz_ct_r[l], wx_r, wy_r, wz_r, tx_ct, ty_ct, tz_ct, grad_step);

				vx_cs_r[l] = tx_cs; vy_cs_r[l] = ty_cs; vz_cs_r[l] = tz_cs;
				vx_ct_r[l] = tx_ct; vy_ct_r[l] = ty_ct; vz_ct_r[l] = tz_ct;

				ReverseField(vx_cs_r[l], vy_cs_r[l], vz_cs_r[l], vx_cs[l], vy_cs[l], vz_cs[l]);
				ReverseField(vx_ct_r[l], vy_ct_r[l], vz_ct_r[l], vx_ct[l], vy_ct[l], vz_ct[l]);
				ReverseField(vx_cs[l], vy_cs[l], vz_cs[l], vx_cs_r[l], vy_cs_r[l], vz_cs_r[l]);
				ReverseField(vx_ct[l], vy_ct[l], vz_ct[l], vx_ct_r[l], vy_ct_r[l], vz_ct_r[l]);

				/*
				sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs_r.vx3d", name, l, ml_max+1, it, dmode, mmode); vx_cs_r[l].save(filename, smode);
				sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs_r.vy3d", name, l, ml_max+1, it, dmode, mmode); vy_cs_r[l].save(filename, smode);
				sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_cs_r.vz3d", name, l, ml_max+1, it, dmode, mmode); vz_cs_r[l].save(filename, smode);
				sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct_r.vx3d", name, l, ml_max+1, it, dmode, mmode); vx_ct_r[l].save(filename, smode);
				sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct_r.vy3d", name, l, ml_max+1, it, dmode, mmode); vy_ct_r[l].save(filename, smode);
				sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_v_ct_r.vz3d", name, l, ml_max+1, it, dmode, mmode); vz_ct_r[l].save(filename, smode);
				//*/

				if (CheckConvergence(energy_list, it+1)) {
					TRACE2("Converged\n");
					it = iter-1;
				}

				if (save_int && it == iter-1) {
				//{
					ComposeFields(vx_cs[l], vy_cs[l], vz_cs[l], xx_cs[l], yy_cs[l], zz_cs[l], hxx_cs, hyy_cs, hzz_cs, 1.0);
					ComposeFields(vx_ct[l], vy_ct[l], vz_ct[l], xx_ct[l], yy_ct[l], zz_ct[l], hxx_ct, hyy_ct, hzz_ct, 1.0);

					ReverseDeformationField(hxx_cs, hyy_cs, hzz_cs, hxx_cs_r, hyy_cs_r, hzz_cs_r);
					ReverseDeformationField(hxx_ct, hyy_ct, hzz_ct, hxx_ct_r, hyy_ct_r, hzz_ct_r);

					// reverse cs and concatenate with ct
					ComposeFields(hxx_cs_r, hyy_cs_r, hzz_cs_r, hxx_ct, hyy_ct, hzz_ct, hxx, hyy, hzz, 1.0);
					// reverse ct and concatenate with cs
					ComposeFields(hxx_ct_r, hyy_ct_r, hzz_ct_r, hxx_cs, hyy_cs, hzz_cs, hxx_r, hyy_r, hzz_r, 1.0);

					sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h.vx3d", name, l, ml_max+1, it, dmode, mmode); hxx.save(filename, smode);
					sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h.vy3d", name, l, ml_max+1, it, dmode, mmode); hyy.save(filename, smode);
					sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h.vz3d", name, l, ml_max+1, it, dmode, mmode); hzz.save(filename, smode);
					sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_r.vx3d", name, l, ml_max+1, it, dmode, mmode); hxx_r.save(filename, smode);
					sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_r.vy3d", name, l, ml_max+1, it, dmode, mmode); hyy_r.save(filename, smode);
					sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_r.vz3d", name, l, ml_max+1, it, dmode, mmode); hzz_r.save(filename, smode);
					//
					sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_cs.vx3d", name, l, ml_max+1, it, dmode, mmode); hxx_cs.save(filename, smode);
					sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_cs.vy3d", name, l, ml_max+1, it, dmode, mmode); hyy_cs.save(filename, smode);
					sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_cs.vz3d", name, l, ml_max+1, it, dmode, mmode); hzz_cs.save(filename, smode);
					sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_ct.vx3d", name, l, ml_max+1, it, dmode, mmode); hxx_ct.save(filename, smode);
					sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_ct.vy3d", name, l, ml_max+1, it, dmode, mmode); hyy_ct.save(filename, smode);
					sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_h_ct.vz3d", name, l, ml_max+1, it, dmode, mmode); hzz_ct.save(filename, smode);

					// warp images
					{
						FVolume vd1t[NumberOfImageChannels];
						FVolume vd2t[NumberOfImageChannels];
						for (i = 0; i < d_num; i++) {
							vd1t[i].allocate(pyrd_vd1[l][i].m_vd_x, pyrd_vd1[l][i].m_vd_y, pyrd_vd1[l][i].m_vd_z);
							vd2t[i].allocate(pyrd_vd2[l][i].m_vd_x, pyrd_vd2[l][i].m_vd_y, pyrd_vd2[l][i].m_vd_z);

							GenerateBackwardWarpVolume(vd1t[i], pyrd_vd1[l][i], hxx_cs, hyy_cs, hzz_cs, 0.0f, false);
							GenerateBackwardWarpVolume(vd2t[i], pyrd_vd2[l][i], hxx_ct, hyy_ct, hzz_ct, 0.0f, false);

							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_vd1tc_%d", name, l, ml_max+1, it, dmode, mmode, i); vd1t[i].save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_vd2tc_%d", name, l, ml_max+1, it, dmode, mmode, i); vd2t[i].save(filename, smode);

							GenerateBackwardWarpVolume(vd1t[i], pyrd_vd1[l][i], hxx_r, hyy_r, hzz_r, 0.0f, false);
							GenerateBackwardWarpVolume(vd2t[i], pyrd_vd2[l][i], hxx, hyy, hzz, 0.0f, false);

							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_vd1t_%d", name, l, ml_max+1, it, dmode, mmode, i); vd1t[i].save(filename, smode);
							sprintf(filename, "%s_[l%d_m%d_s%02d]_[d%02d_m%02d]_vd2t_%d", name, l, ml_max+1, it, dmode, mmode, i); vd2t[i].save(filename, smode);

							vd1t[i].clear();
							vd2t[i].clear();
						}
					}
				}
			}
		}
		//
		TRACE("Update Deformation (Continuous)...done\n");
#endif

		wx.clear(); wy.clear(); wz.clear();
		wx_r.clear(); wy_r.clear(); wz_r.clear();
		tx_cs.clear(); ty_cs.clear(); tz_cs.clear();
		tx_ct.clear(); ty_ct.clear(); tz_ct.clear();

		//exit(0);
	} // l

	{
		ComposeFields(vx_cs[0], vy_cs[0], vz_cs[0], xx_cs[0], yy_cs[0], zz_cs[0], hxx_cs, hyy_cs, hzz_cs, 1.0);
		ComposeFields(vx_ct[0], vy_ct[0], vz_ct[0], xx_ct[0], yy_ct[0], zz_ct[0], hxx_ct, hyy_ct, hzz_ct, 1.0);

		ReverseDeformationField(hxx_cs, hyy_cs, hzz_cs, hxx_cs_r, hyy_cs_r, hzz_cs_r);
		ReverseDeformationField(hxx_ct, hyy_ct, hzz_ct, hxx_ct_r, hyy_ct_r, hzz_ct_r);

		// reverse cs and concatenate with ct
		ComposeFields(hxx_cs_r, hyy_cs_r, hzz_cs_r, hxx_ct, hyy_ct, hzz_ct, hxx, hyy, hzz, 1.0);
		// reverse ct and concatenate with cs
		ComposeFields(hxx_ct_r, hyy_ct_r, hzz_ct_r, hxx_cs, hyy_cs, hzz_cs, hxx_r, hyy_r, hzz_r, 1.0);

		h_x = hxx; h_y = hyy; h_z = hzz;
		h_r_x = hxx_r; h_r_y = hyy_r; h_r_z = hzz_r;
	}

	vx_cs[0].clear(); vy_cs[0].clear(); vz_cs[0].clear();
	vx_ct[0].clear(); vy_ct[0].clear(); vz_ct[0].clear();
	vx_cs_r[0].clear(); vy_cs_r[0].clear(); vz_cs_r[0].clear();
	vx_ct_r[0].clear(); vy_ct_r[0].clear(); vz_ct_r[0].clear();

	for (i = 0; i < d_num; i++) {
		for (k = 0; k < nlevels; k++) {
			pyrd_vd1[k][i].clear();
			pyrd_vd2[k][i].clear();
		}
	}
	for (k = 0; k < nlevels; k++) {
		if (use_tumor_energy) {
			pyrd_tu1[k].clear();
			pyrd_tu2[k].clear();
		}
		pyrd_ab1[k].clear();
		pyrd_ab2[k].clear();
	}
	for (k = 0; k < nlevels; k++) {
		xx_cs[k].clear(); yy_cs[k].clear(); zz_cs[k].clear();
		xx_ct[k].clear(); yy_ct[k].clear(); zz_ct[k].clear();
	}

	TRACE("Update Deformation...done\n");

	return TRUE;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static
inline void GetInv4(double m[16], double inv[16], double* det) {
	inv[ 0] =  m[ 5] * m[10] * m[15] - m[ 5] * m[11] * m[14] - m[ 9] * m[ 6] * m[15] + m[ 9] * m[ 7] * m[14] + m[13] * m[ 6] * m[11] - m[13] * m[ 7] * m[10];
	inv[ 4] = -m[ 4] * m[10] * m[15] + m[ 4] * m[11] * m[14] + m[ 8] * m[ 6] * m[15] - m[ 8] * m[ 7] * m[14] - m[12] * m[ 6] * m[11] + m[12] * m[ 7] * m[10];
	inv[ 8] =  m[ 4] * m[ 9] * m[15] - m[ 4] * m[11] * m[13] - m[ 8] * m[ 5] * m[15] + m[ 8] * m[ 7] * m[13] + m[12] * m[ 5] * m[11] - m[12] * m[ 7] * m[ 9];
	inv[12] = -m[ 4] * m[ 9] * m[14] + m[ 4] * m[10] * m[13] + m[ 8] * m[ 5] * m[14] - m[ 8] * m[ 6] * m[13] - m[12] * m[ 5] * m[10] + m[12] * m[ 6] * m[ 9];
	inv[ 1] = -m[ 1] * m[10] * m[15] + m[ 1] * m[11] * m[14] + m[ 9] * m[ 2] * m[15] - m[ 9] * m[ 3] * m[14] - m[13] * m[ 2] * m[11] + m[13] * m[ 3] * m[10];
	inv[ 5] =  m[ 0] * m[10] * m[15] - m[ 0] * m[11] * m[14] - m[ 8] * m[ 2] * m[15] + m[ 8] * m[ 3] * m[14] + m[12] * m[ 2] * m[11] - m[12] * m[ 3] * m[10];
	inv[ 9] = -m[ 0] * m[ 9] * m[15] + m[ 0] * m[11] * m[13] + m[ 8] * m[ 1] * m[15] - m[ 8] * m[ 3] * m[13] - m[12] * m[ 1] * m[11] + m[12] * m[ 3] * m[ 9];
	inv[13] =  m[ 0] * m[ 9] * m[14] - m[ 0] * m[10] * m[13] - m[ 8] * m[ 1] * m[14] + m[ 8] * m[ 2] * m[13] + m[12] * m[ 1] * m[10] - m[12] * m[ 2] * m[ 9];
	inv[ 2] =  m[ 1] * m[ 6] * m[15] - m[ 1] * m[ 7] * m[14] - m[ 5] * m[ 2] * m[15] + m[ 5] * m[ 3] * m[14] + m[13] * m[ 2] * m[ 7] - m[13] * m[ 3] * m[ 6];   
	inv[ 6] = -m[ 0] * m[ 6] * m[15] + m[ 0] * m[ 7] * m[14] + m[ 4] * m[ 2] * m[15] - m[ 4] * m[ 3] * m[14] - m[12] * m[ 2] * m[ 7] + m[12] * m[ 3] * m[ 6];  
	inv[10] =  m[ 0] * m[ 5] * m[15] - m[ 0] * m[ 7] * m[13] - m[ 4] * m[ 1] * m[15] + m[ 4] * m[ 3] * m[13] + m[12] * m[ 1] * m[ 7] - m[12] * m[ 3] * m[ 5];   
	inv[14] = -m[ 0] * m[ 5] * m[14] + m[ 0] * m[ 6] * m[13] + m[ 4] * m[ 1] * m[14] - m[ 4] * m[ 2] * m[13] - m[12] * m[ 1] * m[ 6] + m[12] * m[ 2] * m[ 5];   
	inv[ 3] = -m[ 1] * m[ 6] * m[11] + m[ 1] * m[ 7] * m[10] + m[ 5] * m[ 2] * m[11] - m[ 5] * m[ 3] * m[10] - m[ 9] * m[ 2] * m[ 7] + m[ 9] * m[ 3] * m[ 6];  
	inv[ 7] =  m[ 0] * m[ 6] * m[11] - m[ 0] * m[ 7] * m[10] - m[ 4] * m[ 2] * m[11] + m[ 4] * m[ 3] * m[10] + m[ 8] * m[ 2] * m[ 7] - m[ 8] * m[ 3] * m[ 6];  
	inv[11] = -m[ 0] * m[ 5] * m[11] + m[ 0] * m[ 7] * m[ 9] + m[ 4] * m[ 1] * m[11] - m[ 4] * m[ 3] * m[ 9] - m[ 8] * m[ 1] * m[ 7] + m[ 8] * m[ 3] * m[ 5];   
	inv[15] =  m[ 0] * m[ 5] * m[10] - m[ 0] * m[ 6] * m[ 9] - m[ 4] * m[ 1] * m[10] + m[ 4] * m[ 2] * m[ 9] + m[ 8] * m[ 1] * m[ 6] - m[ 8] * m[ 2] * m[ 5];

#ifndef USE_SUM_EPSS
	*det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
#else
	*det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12] + epss;
#endif
}

static
inline void ComputeLikelihood4(double inv[16], double c1, double c2, double ym[4], double* like) {
	double ySIy;

	ySIy = ym[0] * (inv[ 0] * ym[0] + inv[ 1] * ym[1] + inv[ 2] * ym[2] + inv[ 3] * ym[3]) +
	       ym[1] * (inv[ 4] * ym[0] + inv[ 5] * ym[1] + inv[ 6] * ym[2] + inv[ 7] * ym[3]) +
	       ym[2] * (inv[ 8] * ym[0] + inv[ 9] * ym[1] + inv[10] * ym[2] + inv[11] * ym[3]) +
	       ym[3] * (inv[12] * ym[0] + inv[13] * ym[1] + inv[14] * ym[2] + inv[15] * ym[3]);

	*like = c1 * exp (c2 * ySIy);
}

//#define USE_TEST_COST
BOOL ComputePosteriors(FVolume* vd, BVolume& mask, FVolume* priors, double (*mv)[4], double (*vv)[16], 
#ifdef USE_PROB_PRIOR
	FVolume* probs, double prob_weight, 
#endif
	FVolume* likelihoods, FVolume* posteriors)
{
	int i, l, m, n;
    //double cost = 0.0;
#ifdef USE_TEST_COST
    int number_of_pixels = 0;
#ifdef USE_PROB_PRIOR
	double cost_prior = 0.0;
#endif
#endif
#ifdef USE_FAST_LIKELIHOOD
	double vv_inv[NumberOfMovingChannels][16];
	double vv_det[NumberOfMovingChannels], vv_c1[NumberOfMovingChannels], vv_c2[NumberOfMovingChannels];
	int vd_x, vd_y, vd_z;

	// assume 4 channels for image
	if (NumberOfFixedChannels != 4) {
		std::cout << "NumberOfFixedChannels must be 4 to use fast likelihood" << std::endl;
		return FALSE;
	}

	vd_x = vd[0].m_vd_x;
	vd_y = vd[0].m_vd_y;
	vd_z = vd[0].m_vd_z;

	for (i = 0; i < NumberOfMovingChannels; i++) {
		GetInv4(vv[i], vv_inv[i], &vv_det[i]);
		vv_c1[i] = 1.0 / (PI2M4 * vcl_sqrt(vv_det[i]));
#ifndef USE_SUM_EPSS
		if (vv_det[i] < eps) {
			vv_c2[i] = 0;
		} else {
			vv_c2[i] = -0.5 / vv_det[i];
		}
#else
		vv_c2[i] = -0.5 / vv_det[i];
#endif
	}
#endif
	//
#ifdef USE_LIKE_TU_MAX
	bool bValidNCR = true;
	bool bValidNE = true;
	if (mv[TU][0] == mv[NCR][0] && mv[TU][1] == mv[NCR][1] && mv[TU][2] == mv[NCR][2] && mv[TU][3] == mv[NCR][3]) {
		bValidNCR = false;
	}
	if (mv[NE][0] == mv[NCR][0] && mv[NE][1] == mv[NCR][1] && mv[NE][2] == mv[NCR][2] && mv[NE][3] == mv[NCR][3]) {
		bValidNE = false;
	}
#endif

	// looping through all voxels
#ifdef USE_TEST_COST
#ifdef USE_PROB_PRIOR
	#pragma omp parallel for private(l,m,n,i) shared(cost,cost_prior,number_of_pixels)
#else
	#pragma omp parallel for private(l,m,n,i) shared(cost,number_of_pixels)
#endif
#else
	#pragma omp parallel for private(l,m,n,i)
#endif
	for (n = 0; n < vd_z; n++) {
		for (m = 0; m < vd_y; m++) {
			for (l = 0; l < vd_x; l++) {
				double prior[NumberOfMovingChannels];
				double like[NumberOfMovingChannels];
				double post[NumberOfMovingChannels];
				double y[4], ym[4];

#ifdef USE_TEST_COST
				#pragma omp atomic
				number_of_pixels++;
#endif

				// making a vnl vector from fixed images
				for (i = 0; i < NumberOfFixedChannels; i++) {
					y[i] = vd[i].m_pData[n][m][l][0];
				}

				// read priors
				for (i = 0; i < NumberOfMovingChannels; i++) {
					prior[i] = priors[i].m_pData[n][m][l][0];
				}

				// computing likelihood values
#ifndef USE_COST_BG
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				if (mask.m_pData[n][m][l][0] > 0 || (prior[TU]+prior[NCR]+prior[NE]) > 0.01) {
#else
				//if (mask.m_pData[n][m][l][0] > 0) {
				if (mask.m_pData[n][m][l][0] > 0 || (prior[TU]+prior[NCR]) > 0.01) {
#endif
#else
				if (1) {
#endif
					// to the number of classes
					for (i = 0; i < NumberOfMovingChannels; i++) {
						ym[0] = y[0] - mv[i][0]; 
						ym[1] = y[1] - mv[i][1]; 
						ym[2] = y[2] - mv[i][2]; 
						ym[3] = y[3] - mv[i][3]; 
#ifdef USE_LIKE_TU_T1CE
						if (i == TU) {
							ComputeLikelihood1(vv[i][5], ym[1], &like[i]);
						} else {
							//ComputeLikelihood4(vv[i], ym, &like[i]);
							ComputeLikelihood4(vv_inv[i], vv_c1[i], vv_c2[i], ym, &like[i]);
						}
#else
						//ComputeLikelihood4(vv[i], ym, &like[i]);
						ComputeLikelihood4(vv_inv[i], vv_c1[i], vv_c2[i], ym, &like[i]);
#endif
					}
#ifndef USE_LIKE_TU_MAX
					// compute the denum
					double denum = epss;
					for (i = 0; i < NumberOfMovingChannels; i++) {
						denum += prior[i] * like[i];
					}
					// compute the posterior
					for (i = 0; i < NumberOfMovingChannels; i++) {
						post[i] = prior[i] * like[i] / denum;
					}
#else
					// compute the denum
					double denum = epss;
					double like_tu_max = max(like[TU], max(like[NCR], like[NE]));
					double post_tu_max, denum_tu;
					for (i = 0; i < NumberOfMovingChannels; i++) {
						if (i != TU && i != NCR && i != NE) {
							denum += prior[i] * like[i];
						}
					}
					denum += prior[TU] * 3 * like_tu_max;
					// compute the posterior
					if (denum != 0) {
						for (i = 0; i < NumberOfMovingChannels; i++) {
							if (i != TU && i != NCR && i != NE) {
								post[i] = prior[i] * like[i] / denum;
							}
						}
						post_tu_max = prior[TU] * 3 * like_tu_max / denum;
						//
						denum_tu = like[TU];
						if (bValidNCR) {
							denum_tu += like[NCR];
						}
						if (bValidNE) {
							denum_tu += like[NE];
						}
						if (denum_tu != 0) {
							post[TU] = post_tu_max * like[TU] / denum_tu;
							if (bValidNCR) {
								post[NCR] = post_tu_max * like[NCR] / denum_tu;
							} else {
								post[NCR] = post[TU];
							}
							if (bValidNE) {
								post[NE] = post_tu_max * like[NE] / denum_tu;
							} else {
								post[NE] = post[NCR];
							}
						} else {
							post[TU] = 0;
							post[NCR] = 0;
							post[NE] = 0;
						}
					} else {
						for (i = 0; i < NumberOfMovingChannels; i++) {
							post[i] = 0.0;
						}
					}
#endif
				} else {
					// compute the posterior
					for (i = 0; i < NumberOfMovingChannels; i++) {
						post[i] = 0.0;
						like[i] = 1.0;
					}
#ifdef USE_WARPED_BG
					post[BG] = 1.0;
#endif
				}
				for (i = 0; i < NumberOfMovingChannels; i++) {
					posteriors[i].m_pData[n][m][l][0] = post[i];
				}
				if (likelihoods != NULL) {
					for (i = 0; i < NumberOfMovingChannels; i++) {
						likelihoods[i].m_pData[n][m][l][0] = like[i];
					}
				}

#ifdef USE_TEST_COST
				for (i = 0; i < NumberOfMovingChannels; i++) {
					// we have to take care of negative values
					double val = like[i] * prior[i];
					//if (val < epss) val = epss;
					if (val <= 0.0) val = epss;
					#pragma omp atomic
					cost -= post[i] * log(val);
				}
#ifdef USE_PROB_PRIOR
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				#pragma omp atomic
				cost_prior += prob_weight * 9 * (prior[TU] - probs[TU].m_pData[n][m][l][0]) * (prior[TU] - probs[TU].m_pData[n][m][l][0]);
#else
				// multiplying 4 is considering tu_prior = prior[TU] + prior[NCR]
				#pragma omp atomic
				cost_prior += prob_weight * 4 * (prior[TU] - probs[TU].m_pData[n][m][l][0]) * (prior[TU] - probs[TU].m_pData[n][m][l][0]);
				//cost_prior += prob_weight * (post[TU] + post[NCR]) * ((prior[TU] - probs[TU].m_pData[n][m][l][0]) * (prior[TU] - probs[TU].m_pData[n][m][l][0]));
#endif
#endif
#endif
			}
		}
	}

#ifdef USE_TEST_COST
	/*
#if 1
	m_SumOfEMLogs = cost;
#ifdef USE_PROB_PRIOR
	m_SumOfEMLogs += cost_prior;
#endif
	m_Metric = m_SumOfEMLogs / number_of_pixels; 
#endif
	//*/

	cost /= number_of_pixels;
#ifdef USE_PROB_PRIOR
	cost_prior /= number_of_pixels;
#endif

#ifdef USE_PROB_PRIOR
	TRACE2("cost = %f, cost_prior = %f, sum = %f, number_of_pixels = %d\n", cost, cost_prior, cost + cost_prior, number_of_pixels);
#else
	TRACE2("cost = %f, number_of_pixels = %d\n", cost, number_of_pixels);
#endif
#endif

	return TRUE;
}

//#define USE_COMPUTEMSMAP_TRACE
BOOL ComputeMSMap(FVolume* vd, BVolume& mask, BVolume& label_map, FVolume* ms_map, double (*mv_init)[4], int* rl, int rl_num, int* cl, int cl_num, int samp = 1) 
{
	char fdata_file_name[1024] = {0,};
	int no_lsh = 0;
	int K, L;
	int	k_neigh = 120;
	int jump = 1;
	double percent = 0.0;
	float width = -1;
	//
	FAMS* fams;
	//
	int vd_x, vd_y, vd_z;
	int l, m, n, i, j, k;
	int **rl_merge, *rl_merge_num;

	vd_x = label_map.m_vd_x;
	vd_y = label_map.m_vd_y;
	vd_z = label_map.m_vd_z;

	rl_merge_num = (int*)MyAlloc(rl_num * sizeof(int));
	rl_merge = (int**)MyAlloc(rl_num * sizeof(int*));
	for (i = 0; i < rl_num; i++) {
		rl_merge[i] = (int*)MyAlloc(rl_num * sizeof(int));
	}
	for (j = 0; j < rl_num; j++) {
		rl_merge_num[j] = 0;
		for (i = 0; i < rl_num; i++) {
			if (i == j) {
				continue;
			}
			BOOL bSame = TRUE;
			for (k = 0; k < NumberOfFixedChannels; k++) {
				if (mv_init[rl[j]][k] != mv_init[rl[i]][k]) {
					bSame = FALSE;
					break;
				}
			}
			if (bSame) {
				rl_merge[j][rl_merge_num[j]] = i;
				rl_merge_num[j]++;
			}
		}
	}

	// init
	fams = new FAMS(no_lsh);

	// load data
	{
		BVolume samp_test;
		float* pttemp = NULL;
		int n_, d_;

		d_ = NumberOfFixedChannels;
		//
		samp_test.allocate(vd_x, vd_y, vd_z);
		pttemp = new float[vd_x * vd_y * vd_z * d_];

		for (n = 0; n < vd_z; n++) {
			for (m = 0; m < vd_y; m++) {
				for (l = 0; l < vd_x; l++) {
					samp_test.m_pData[n][m][l][0] = 0;
				}
			}
		}
		for (n = 0; n < vd_z; n += samp) {
			for (m = 0; m < vd_y; m += samp) {
				for (l = 0; l < vd_x; l += samp) {
					samp_test.m_pData[n][m][l][0] = 1;
				}
			}
		}

		n_ = 0;
		for (n = 0; n < vd_z; n++) {
			for (m = 0; m < vd_y; m++) {
				for (l = 0; l < vd_x; l++) {
					if (samp_test.m_pData[n][m][l][0] == 1) {
						int val = label_map.m_pData[n][m][l][0];
						BOOL bRgn = FALSE;
						for (i = 0; i < rl_num; i++) {
							if (val == label_idx[rl[i]]) {
								bRgn = TRUE;
								break;
							}
						}
						if (bRgn) {
							for (i = 0; i < d_; i++) {
								pttemp[n_*d_+i] = vd[i].m_pData[n][m][l][0];
							}
							n_++;
						}
					}
				}
			}
		}

		fams->CleanPoints();

		fams->n_ = n_;
		fams->d_ = d_;

		// allocate and convert to integer
		for (i = 0, fams->minVal_ = pttemp[0], fams->maxVal_ = pttemp[0]; i < (n_ * d_); i++) {
			if (fams->minVal_ > pttemp[i]) {
				fams->minVal_ = pttemp[i];
			} else if (fams->maxVal_ < pttemp[i]) {
				fams->maxVal_ = pttemp[i];
			}
		}
		fams->data_ = new unsigned short[n_ * d_];
		fams->rr_ = new double[d_];
		fams->hasPoints_ = 1;
		float deltaVal = fams->maxVal_ - fams->minVal_;
		if (deltaVal == 0) deltaVal = 1;
		for (i = 0; i < (n_ * d_); i++) {
			fams->data_[i] = (unsigned short)(65535.0 * (pttemp[i] - fams->minVal_) / deltaVal);
		}
		fams->dataSize_ = d_ * sizeof(unsigned short);

		fams->points_ = new fams_point[n_];
		unsigned short* dtempp;
		for (i = 0, dtempp = fams->data_; i < n_; i++, dtempp += d_)
		{
			fams->points_[i].data_ = dtempp;
			fams->points_[i].usedFlag_ = 0;
		}

		delete [] pttemp;
	}

#if 0
	{
		float epsilon;
		int Kmin, Kjump, Kmax;
		int Lmax;

		Kmin = 10;
		Kmax = 30;
		Kjump = 2;
		Lmax = 50;
		epsilon = 0.05;

		// find K L
		fams->FindKL(Kmin, Kmax, Kjump, Lmax, k_neigh, width, epsilon, K, L);
		TRACE2("Found K = %d L = %d\n", K, L);
	}
#else
	K = 30;
	L = 6;
#ifdef USE_COMPUTEMSMAP_TRACE
	TRACE2("Using K = %d L = %d\n", K, L);
#endif
#endif
#ifdef USE_COMPUTEMSMAP_TRACE
	TRACE2("k_neigh = %d\n", k_neigh);
#endif

	//sprintf(fdata_file_name, "%s%sfams_pilot_%d.txt", tmp_folder, DIR_SEP, k_neigh);
	fams->RunFAMS(K, L, k_neigh, percent, jump, width, fdata_file_name);

	int *prunedlabel, *nnmode;
	float *cl_dist, *rl_dist;
	float dist_min;
	int dist_min_idx;

	prunedlabel = (int*)MyAlloc(fams->npm_ * sizeof(int));
	nnmode = (int*)MyAlloc(rl_num * sizeof(int));
	cl_dist = (float*)MyAlloc(cl_num * sizeof(float));
	rl_dist = (float*)MyAlloc(fams->npm_ * sizeof(float));

	for (i = 0; i < fams->npm_; i++) {
		for (k = 0; k < cl_num; k++) {
			cl_dist[k] = 0;
		}
		for (j = 0; j < fams->d_; j++) {
			float val = fams->prunedmodes_[i * fams->d_ + j] * (fams->maxVal_ - fams->minVal_) / 65535.0 + fams->minVal_;
			for (k = 0; k < cl_num; k++) {
				cl_dist[k] += (val - mv_init[cl[k]][j]) * (val - mv_init[cl[k]][j]);
			}
		}
		//
		dist_min = cl_dist[0];
		dist_min_idx = 0;
		for (k = 1; k < cl_num; k++) {
			if (dist_min > cl_dist[k]) {
				dist_min = cl_dist[k];
				dist_min_idx = k;
			}
		}
		//
#ifdef USE_COMPUTEMSMAP_TRACE
		TRACE2("mode %d (%d): ", i, fams->nprunedmodes_[i]);
		for (k = 0; k < cl_num; k++) {
			if (k == dist_min_idx) {
				TRACE2("(dist[%s] = %g) ", label[cl[k]], cl_dist[k]);
			} else {
				TRACE2("dist[%s] = %g ", label[cl[k]], cl_dist[k]);
			}
		}
		TRACE2("\n");
#endif
		//
		// -1 means outlier label
		prunedlabel[i] = -1;
		for (k = 0; k < rl_num; k++) {
			if (cl[dist_min_idx] == rl[k]) {
				prunedlabel[i] = k;
				break;
			}
		}
	}
	for (k = 0; k < rl_num; k++) {
		for (i = 0; i < fams->npm_; i++) {
			rl_dist[i] = 0;
			for (j = 0; j < fams->d_; j++) {
				float val = fams->prunedmodes_[i * fams->d_ + j] * (fams->maxVal_ - fams->minVal_) / 65535.0 + fams->minVal_;
				rl_dist[i] += (val - mv_init[rl[k]][j]) * (val - mv_init[rl[k]][j]);
			}
		}
		//
		dist_min = rl_dist[0];
		dist_min_idx = 0;
		for (i = 1; i < fams->npm_; i++) {
			if (dist_min > rl_dist[i]) {
				dist_min = rl_dist[i];
				dist_min_idx = i;
			}
		}
		nnmode[k] = dist_min_idx;
		//
#ifdef USE_COMPUTEMSMAP_TRACE
		TRACE2("%s: ", label[rl[k]]);
		for (i = 0; i < fams->npm_; i++) {
			if (i == dist_min_idx) {
				TRACE2("([%d] %g) ", i, rl_dist[i]);
			} else {
				TRACE2("[%d] %g ", i, rl_dist[i]);
			}
		}
		TRACE2("\n");
#endif
	}

	/*
	// save the data
	sprintf(fdata_file_name, "jsr_fams_out_%d.txt", this->GetNumberOfElapsedIterations());
	fams->SaveModes(fdata_file_name);
	//*/
	/*
	// save pruned modes modes
	sprintf(fdata_file_name, "jsr_fams_modes_%d.txt", this->GetNumberOfElapsedIterations());
	fams->SavePrunedModes(fdata_file_name);
	//*/

	{
		int idx = 0;
		for (n = 0; n < vd_z; n++) {
			for (m = 0; m < vd_y; m++) {
				for (l = 0; l < vd_x; l++) {
					for (i = 0; i < rl_num; i++) {
						ms_map[rl[i]].m_pData[n][m][l][0] = 0;
					}
				}
			}
		}
		for (n = 0; n < vd_z; n += samp) {
			for (m = 0; m < vd_y; m += samp) {
				for (l = 0; l < vd_x; l += samp) {
					int val = label_map.m_pData[n][m][l][0];
					BOOL bRgn = FALSE;
					for (i = 0; i < rl_num; i++) {
						if (val == label_idx[rl[i]]) {
							bRgn = TRUE;
							break;
						}
					}
					if (bRgn) {
						for (i = 0; i < fams->npm_; i++) {
							if (fams->modes_[idx  ] == fams->prunedmodes_[i*fams->d_  ] &&
								fams->modes_[idx+1] == fams->prunedmodes_[i*fams->d_+1] &&
								fams->modes_[idx+2] == fams->prunedmodes_[i*fams->d_+2] &&
								fams->modes_[idx+3] == fams->prunedmodes_[i*fams->d_+3]) {
								int pl = prunedlabel[i];
								if (pl < 0) {
									// label matched to outliers
								} else {
									ms_map[rl[pl]].m_pData[n][m][l][0] = 1;
									for (k = 0; k < rl_merge_num[pl]; k++) {
										ms_map[rl[rl_merge[pl][k]]].m_pData[n][m][l][0] = 1;
									}
									for (k = 0; k < rl_num; k++) {
										if (nnmode[k] == i) {
											ms_map[rl[k]].m_pData[n][m][l][0] = 1;
										}
									}
								}
								break;
							}
						}
						if (i == fams->npm_) {
							// error
						}
						//
						idx += fams->d_;
					}
				}
			}
		}

		MyFree(prunedlabel);
		MyFree(nnmode);
		MyFree(cl_dist);
		MyFree(rl_dist);
	}

	for (i = 0; i < rl_num; i++) {
		MyFree(rl_merge[i]);
	}
	MyFree(rl_merge);
	MyFree(rl_merge_num);

	delete fams;

	return TRUE;
}

BOOL UpdateMeansAndVariances(FVolume* vd, BVolume& mask, FVolume* posteriors, double (*mv_)[4], double (*vv_)[16], double (*mv_init_)[4], int NumberOfElapsedIterations, bool bMeanShiftUpdate, bool mean_shift_update, bool ms_tumor, bool ms_edema)
{
	// define a vector object to keep sum of weighted vectors for each class
	double mv_sum[NumberOfMovingChannels][NumberOfFixedChannels];
	// define a vector object to keep sum of weighted outer products for each class
	double vv_sum[NumberOfMovingChannels][NumberOfFixedChannels*NumberOfFixedChannels];
	// define a vector object to keep sum of the weights for each class
	double w_sum[NumberOfMovingChannels];
	double mv[NumberOfMovingChannels][NumberOfFixedChannels];
	int i, j, k, l, m, n;
	//
	int vd_x = vd[0].m_vd_x;
	int vd_y = vd[0].m_vd_y;
	int vd_z = vd[0].m_vd_z;
	//
#ifdef USE_OUTLIER_REJECTION_LAMBDA
	double p_vv[NumberOfMovingChannels][16];
	double p_vv_inv[NumberOfMovingChannels][16];
	double p_vv_det[NumberOfMovingChannels], p_vv_c1[NumberOfMovingChannels], p_vv_c2[NumberOfMovingChannels];
	double p_mv[NumberOfMovingChannels][4];
	double p_vv_lambda[NumberOfMovingChannels];
	double p_vv_k = 3.5;
	double p_like_mean[NumberOfMovingChannels];
	int p_like_num[NumberOfMovingChannels];
	bool use_like = false;

#ifndef USE_FAST_LIKELIHOOD
	std::cout << "not implemented" << std::endl;
	return;
#endif
	// assume 4 channels for image
	if (NumberOfFixedChannels != 4) {
		std::cout << "NumberOfFixedChannels must be 4 to use fast likelihood" << std::endl;
		return;
	}

	for (i = 0; i < NumberOfMovingChannels; i++) {
		for (j = 0; j < 4; j++) {
			for (k = 0; k < 4; k++) {
				p_vv[i][j*4+k] = vv_[i](j, k);
			}
			p_mv[i][j] = mv_[i](j);
		}
	}
	for (i = 0; i < NumberOfMovingChannels; i++) {
		GetInv4(p_vv[i], p_vv_inv[i], &p_vv_det[i]);
		p_vv_c1[i] = 1.0 / (PI2M4 * vcl_sqrt(p_vv_det[i]));
#ifndef USE_SUM_EPSS
		if (p_vv_det[i] < eps) {
			p_vv_c2[i] = 0;
		} else {
			p_vv_c2[i] = -0.5 / p_vv_det[i];
		}
#else
		p_vv_c2[i] = -0.5 / p_vv_det[i];
#endif
		//
		if (NumberOfElapsedIterations) {
			p_vv_lambda[i] = exp(-0.5 * p_vv_k * p_vv_k) / (PI2M4 * vcl_sqrt(p_vv_det[i]));
			//
			printf("%s: lambda = %e\n", label[i], p_vv_lambda[i]);
		} else {
			p_vv_lambda[i] = eps;
		}
		//
		p_like_mean[i] = 0;
		p_like_num[i] = 0;
	}
	if (NumberOfElapsedIterations) {
		use_like = true;
	}
#endif

#ifdef USE_OUTLIER_REJECTION_INIT_MEANS
	FVolume dist_map[NumberOfMovingChannels];

	{
		int* dist_hist[NumberOfMovingChannels];
		int dist_hist_n[NumberOfMovingChannels];
		int dist_th[NumberOfMovingChannels];
		int hist_max = 255*255*4;
		double mv_init[NumberOfMovingChannels][NumberOfFixedChannels];
		double p_th, d_th;

		if (this->GetNumberOfElapsedIterations()) {
			 p_th = 0.5;
			 d_th = 0.7;
		} else {
			 p_th = 0.1;
			 d_th = 0.3;
		}

		for (i = 0; i < NumberOfMovingChannels; i++) {
			dist_map[i].allocate(vd_x, vd_y, vd_z);
		}

		for (k = 0; k < NumberOfMovingChannels; k++) {
			for (i = 0; i < NumberOfFixedChannels; i++) {
				mv_init[k][i] = mv_[k][i];
			}
		}

		for (i = 0; i < NumberOfMovingChannels; i++) {
			dist_hist_n[i] = 0;
			dist_hist[i] = (int*)malloc((hist_max+1) * sizeof(int));
			for (j = 0; j <= hist_max; j++) {
				dist_hist[i][j] = 0;
			}
		}

		for (n = 0; n < vd_z; n++) {
			for (m = 0; m < vd_y; m++) {
				for (l = 0; l < vd_x; l++) {
					double y[NumberOfFixedChannels];

					for (i = 0; i < NumberOfMovingChannels; i++) {
						dist_map[i].m_pData[n][m][l][0] = hist_max;
					}
					//
					for (i = 0; i < NumberOfFixedChannels; i++) {
						y[i] = vd[i].m_pData[n][m][l][0];
					}
					if (mask.m_pData[n][m][l][0] > 0) {
						for (i = 0; i < NumberOfMovingChannels; i++) {
							if ((double)posteriors[i].m_pData[n][m][l][0] > p_th) {
								int val, dist = 0;
								for (j = 0; j < NumberOfFixedChannels; j++) {
									val = y[j] - mv_init[i][j];
									dist += val * val;
								}
								dist = min(dist, hist_max);
								dist_map[i].m_pData[n][m][l][0] = dist;
								dist_hist[i][dist]++;
								dist_hist_n[i]++;
							}
						}
					}
				}
			}
		}

		for (i = 0; i < NumberOfMovingChannels; i++) {
			int dist_hist_th = (int)(dist_hist_n[i] * d_th);
			int acc = 0;
			for (j = 0; j < hist_max; j++) {
				acc += dist_hist[i][j];
				if (acc >= dist_hist_th) {
					dist_th[i] = j;
					break;
				}
			}

			printf("%s: dist_th = %d, dist_hist_n = %d\n", label[i], dist_th[i], dist_hist_n[i]);
		}

		/*
		for (i = 0; i < NumberOfMovingChannels; i++) {
			char name[1024];
			sprintf(name,"jsr_dist_map_%d_%d.nii.gz", this->GetNumberOfElapsedIterations(), i);
			dist_map[i].save(name, 1);
		}
		//*/

		for (n = 0; n < vd_z; n++) {
			for (m = 0; m < vd_y; m++) {
				for (l = 0; l < vd_x; l++) {
					for (i = 0; i < NumberOfMovingChannels; i++) {
						if (0
#ifdef USE_INIT_MEANS_CSF
							|| i == CSF
#endif
#ifdef USE_INIT_MEANS_VS
							|| i == VS
#endif
#ifdef USE_INIT_MEANS_VT
#if defined(USE_9_PRIORS)
							|| i == VT
#endif
#if defined(USE_10_PRIORS) || defined(USE_11_PRIORS)
							|| i == VT
#endif
#endif
#ifdef USE_INIT_MEANS_ED
							|| i == ED
#endif
#ifdef USE_INIT_MEANS_GM
							|| i == GM
#endif
#ifdef USE_INIT_MEANS_WM
							|| i == WM
#endif
						) {
							if (dist_map[i].m_pData[n][m][l][0] < dist_th[i]) {
								dist_map[i].m_pData[n][m][l][0] = 1;
							} else {
								dist_map[i].m_pData[n][m][l][0] = 0;
							}
#ifdef USE_WM_UPDATE_SKIP_ED_REGION
						} else if (i == WM) {
							if ((double)posterior[ED].m_pData[n][m][l][0] > 0) {
								dist_map[i].m_pData[n][m][l][0] = 0;
							} else {
								dist_map[i].m_pData[n][m][l][0] = 1;
							}
#endif
						} else {
							dist_map[i].m_pData[n][m][l][0] = 1;
						}
					}
				}
			}
		}

		/*
		for (i = 0; i < NumberOfMovingChannels; i++) {
			char name[1024];
			sprintf(name, "jsr_dist_map_th_%d_%d.nii.gz", this->GetNumberOfElapsedIterations(), i);
			dist_map[i].save(name, 1);
		}
		//*/

		for (i = 0; i < NumberOfMovingChannels; i++) {
			free(dist_hist[i]);
		}
	}
#endif

	FVolume ms_map[NumberOfMovingChannels];

	for (i = 0; i < NumberOfMovingChannels; i++) {
		ms_map[i].allocate(vd_x, vd_y, vd_z);
	}
	for (n = 0; n < vd_z; n++) {
		for (m = 0; m < vd_y; m++) {
			for (l = 0; l < vd_x; l++) {
				for (i = 0; i < NumberOfMovingChannels; i++) {
					ms_map[i].m_pData[n][m][l][0] = 1;
				}
			}
		}
	}

	if (mean_shift_update && bMeanShiftUpdate) {
		if (NumberOfElapsedIterations) {
			BVolume label_map;
			long long size_tu = 0;
			long long size_wm_ed = 0;

			label_map.allocate(vd_x, vd_y, vd_z);

			for (n = 0; n < vd_z; n++) {
				for (m = 0; m < vd_y; m++) {
					for (l = 0; l < vd_x; l++) {
						float max_l_val = posteriors[0].m_pData[n][m][l][0];
						int max_l_idx = 0;
						for (i = 1; i < NumberOfMovingChannels; i++) {
							if (max_l_val < posteriors[i].m_pData[n][m][l][0]) {
								max_l_val = posteriors[i].m_pData[n][m][l][0];
								max_l_idx = i;
							}
						}
						label_map.m_pData[n][m][l][0] = label_idx[max_l_idx];

#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
						if (max_l_idx == TU || max_l_idx == NCR || max_l_idx == NE) {
#else
						if (max_l_idx == TU || max_l_idx == NCR) {
#endif
							size_tu++;
						}
						if (max_l_idx == WM || max_l_idx == ED) {
							size_wm_ed++;
						}
					}
				}
			}

			{
				int rl[NumberOfMovingChannels];
				int cl[NumberOfMovingChannels];
				int rl_num, cl_num;
				//
				if (ms_tumor) {
					rl[0] = TU;
					rl[1] = NCR;
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
					rl[2] = NE;
					rl_num = 3;
#else
					rl_num = 2;
#endif
					cl[0] = TU;
					cl[1] = NCR;
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
					cl[2] = NE;
					cl_num = 3;
					//cl[3] = ED;
					//cl_num = 4;
#else
					cl[2] = ED;
					cl_num = 3;
#endif
					TRACE2("size_tu = %d\n", size_tu);
					if (size_tu > 800000) {
						TRACE2("size_tu > 800000: skip ComputeMSMap\n");
					} else if (size_tu > 100000) {
						ComputeMSMap(vd, mask, label_map, ms_map, mv_init_, rl, rl_num, cl, cl_num, 2);
					} else {
						ComputeMSMap(vd, mask, label_map, ms_map, mv_init_, rl, rl_num, cl, cl_num, 1);
					}
				}
				//
				if (ms_edema) {
					rl[0] = WM;
					rl[1] = ED;
					rl_num = 2;
					cl[0] = WM;
					cl[1] = ED;
					cl_num = 2;
					TRACE2("size_wm_ed = %d\n", size_wm_ed);
					if (size_wm_ed > 800000) {
						TRACE2("size_wm_ed > 800000: skip ComputeMSMap\n");
					} else if (size_wm_ed > 100000) {
						ComputeMSMap(vd, mask, label_map, ms_map, mv_init_, rl, rl_num, cl, cl_num, 2);
					} else { 
						ComputeMSMap(vd, mask, label_map, ms_map, mv_init_, rl, rl_num, cl, cl_num, 1);
					}
					//*
					// reset WM map
					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								ms_map[WM].m_pData[n][m][l][0] = 1;
							}
						}
					}
					//*/
				}
				//*/
				//
				/*
				{
					rl[0] = GM;
					rl[1] = WM;
					rl[2] = ED;
					rl_num = 3;
					cl[0] = GM;
					cl[1] = WM;
					cl[2] = ED;
					cl_num = 3;
					ComputeMSMap(f_it_vect, mask, label_map, ms_map, rl, rl_num, cl, cl_num, 2);
				}
				//*/
				//
				/*
				{
					rl[0] = GM;
					rl_num = 1;
					cl[0] = GM;
					cl[1] = WM;
					cl[2] = ED;
					cl_num = 3;
					ComputeMSMap(f_it_vect, mask, label_map, ms_map, rl, rl_num, cl, cl_num, 2);
				}
				//*/
				//
				/*
				{
					rl[0] = CSF;
					rl[1] = VS;
					rl_num = 2;
					cl[0] = CSF;
					cl[1] = VS;
					cl_num = 2;
					ComputeMSMap(f_it_vect, mask, label_map, ms_map, rl, rl_num, cl, cl_num, 1);
				}
				//*/
			}

			/*
			for (i = 0; i < NumberOfMovingChannels; i++) {
				char name[1024];
				sprintf(name, "jsr_ms_map_th_%d_%d.nii.gz", this->GetNumberOfElapsedIterations(), i);
				ms_map[i].save(name, 1);
			}
			//*/

			label_map.clear();
		}
	}

	for (k = 0; k < NumberOfMovingChannels; k++) {
		w_sum[k] = eps;
		for (j = 0; j < NumberOfFixedChannels; j++) {
			mv_sum[k][j] = eps;
			for (i = 0; i < NumberOfFixedChannels; i++) {
				if (i == j) {
					vv_sum[k][NumberOfFixedChannels*j+i] = eps;
				} else {
					vv_sum[k][NumberOfFixedChannels*j+i] = 0;
				}
			}
		}
	}

	// sum loop for means
	for (n = 0; n < vd_z; n++) {
		for (m = 0; m < vd_y; m++) {
			for (l = 0; l < vd_x; l++) {
				double y[NumberOfFixedChannels];
#ifdef USE_OUTLIER_REJECTION_LAMBDA
				double p_ym[NumberOfFixedChannels];
				double p_like[NumberOfMovingChannels];
#endif
				//
				for (i = 0; i < NumberOfFixedChannels; i++) {
					y[i] = vd[i].m_pData[n][m][l][0];
				}
				if (mask.m_pData[n][m][l][0] > 0) {
					for (k = 0; k < NumberOfMovingChannels; k++) {
						double p;
						//
#ifdef USE_OUTLIER_REJECTION_INIT_MEANS
						if (dist_map[k].m_pData[n][m][l][0] == 0) {
							continue;
						}
#endif
						//
						if (ms_map[k].m_pData[n][m][l][0] == 0) {
							continue;
						}
						//
						p = (double)posteriors[k].m_pData[n][m][l][0];
						//
						if (p > 0) {
#ifdef USE_OUTLIER_REJECTION_LAMBDA
							if (use_like
								&& k != BG
#if defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								&& k != CB
#endif
								) {
									p_ym[0] = y[0] - p_mv[k][0]; 
									p_ym[1] = y[1] - p_mv[k][1]; 
									p_ym[2] = y[2] - p_mv[k][2]; 
									p_ym[3] = y[3] - p_mv[k][3]; 
									ComputeLikelihood4(p_vv_inv[k], p_vv_c1[k], p_vv_c2[k], p_ym, &p_like[k]);
									p *= p_like[k] / (p_like[k] + p_vv_lambda[k]);
									//
									p_like_mean[k] += p_like[k];
									p_like_num[k]++;
							}
#endif
							//
							for (i = 0; i < NumberOfFixedChannels; i++) {
								mv_sum[k][i] += p * y[i];
							}
							w_sum[k] += p;
						}
					}
				}
			}
		}
	}
#ifdef USE_OUTLIER_REJECTION_LAMBDA
	for (i = 0; i < NumberOfMovingChannels; i++) {
		if (p_like_num[i] != 0) {
			p_like_mean[i] /= p_like_num[i];
		} else {
			p_like_mean[i] = 0;
		}
		//
		printf("%s: p_like_mean = %e, p_like_num = %d\n", label[i], p_like_mean[i], p_like_num[i]);
	}
#endif

	// computing updated means to the number of classes
	if (NumberOfElapsedIterations) {
		for (k = 0; k < NumberOfMovingChannels; k++) {
			double w_1 = 1.0 / (w_sum[k] + eps);
			for (i = 0; i < NumberOfFixedChannels; i++) {
				mv[k][i] = w_1 * mv_sum[k][i] + eps;
				mv_[k][i] = mv[k][i];
			}
		}
	} else {
		// in the first iteration we want to favor user supplied values
		for (k = 0; k < NumberOfMovingChannels; k++) {
			double w_1 = 1.0 / (w_sum[k] + eps);
			for (i = 0; i < NumberOfFixedChannels; i++) {
				mv[k][i] = w_1 * mv_sum[k][i] + eps;
				if (mv_[k][i] == INITIAL_MEAN_VALUE) {
					mv_[k][i] = mv[k][i];
				} else {
					mv[k][i] = mv_[k][i];
				}
			}
		}
	}

	// sum loop for variances
	for (n = 0; n < vd_z; n++) {
		for (m = 0; m < vd_y; m++) {
			for (l = 0; l < vd_x; l++) {
				double y[NumberOfFixedChannels];
				double ym[NumberOfFixedChannels];
#ifdef USE_OUTLIER_REJECTION_LAMBDA
				double p_ym[NumberOfFixedChannels];
				double p_like[NumberOfMovingChannels];
#endif
				//
				for (i = 0; i < NumberOfFixedChannels; i++) {
					y[i] = vd[i].m_pData[n][m][l][0];
				}
				if (mask.m_pData[n][m][l][0] > 0) {
					for (k = 0; k < NumberOfMovingChannels; k++) {
						double p;
						//
#ifdef USE_OUTLIER_REJECTION_INIT_MEANS
						if (dist_map[k].m_pData[n][m][l][0] == 0) {
							continue;
						}
#endif
						//
						if (ms_map[k].m_pData[n][m][l][0] == 0) {
							continue;
						}
						//
						p = (double)posteriors[k].m_pData[n][m][l][0];
						//
						if (p > 0) {
#ifdef USE_OUTLIER_REJECTION_LAMBDA
							if (use_like
								&& k != BG
#if defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								&& k != CB
#endif
								) {
									p_ym[0] = y[0] - p_mv[k][0]; 
									p_ym[1] = y[1] - p_mv[k][1]; 
									p_ym[2] = y[2] - p_mv[k][2]; 
									p_ym[3] = y[3] - p_mv[k][3]; 
									ComputeLikelihood4(p_vv_inv[k], p_vv_c1[k], p_vv_c2[k], p_ym, &p_like[k]);
									p *= p_like[k] / (p_like[k] + p_vv_lambda[k]);
							}
#endif
							//
							for (i = 0; i < NumberOfFixedChannels; i++) {
								ym[i] = y[i] - mv[k][i];
							}
							for (j = 0; j < NumberOfFixedChannels; j++) {
								for (i = 0; i < NumberOfFixedChannels; i++) {
									vv_sum[k][NumberOfFixedChannels*j+i] += p * ym[j] * ym[i];
								}
							}
						}
					}
				}
			}
		}
	}

	// computing updated variances to the number of classes
	for (k = 0; k < NumberOfMovingChannels; k++) {
		double w_1 = 1.0 / (w_sum[k] + eps);
		for (j = 0; j < NumberOfFixedChannels; j++) {
			for (i = 0; i < NumberOfFixedChannels; i++) {
				vv_[k][NumberOfFixedChannels*j+i] = w_1 * vv_sum[k][NumberOfFixedChannels*j+i];
			}
		}
	}

#ifdef USE_OUTLIER_REJECTION_INIT_MEANS
	for (i = 0; i < NumberOfMovingChannels; i++) {
		dist_map[i].clear();
	}
#endif
	for (i = 0; i < NumberOfMovingChannels; i++) {
		ms_map[i].clear();
	}

	//*
	TRACE2("updated means and variances:\n");
	for (k = 0; k < NumberOfMovingChannels; k++) {
		TRACE2("%3s:", label[k]);
		for (i = 0; i < NumberOfFixedChannels; i++) {
			TRACE2(" %7.2f", mv_[k][i]);
		}
		TRACE2(":");
		for (i = 0; i < NumberOfFixedChannels; i++) {
			TRACE2(" %8.2f", vv_[k][i * NumberOfFixedChannels + i]);
		}
		TRACE2(": %f", w_sum[k]);
		TRACE2("\n");
	}
	//*/

	return TRUE;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
