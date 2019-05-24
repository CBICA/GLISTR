/**
 * @file  itkJointSegmentationRegistrationFunction.txx
 * @brief Function class implementing the EM based joint segmentation-registration algorithm.
 *
 * Copyright (c) 2011-2014 University of Pennsylvania. All rights reserved.<br />
 * See http://www.cbica.upenn.edu/sbia/software/license.html or COPYING file.
 *
 * Contact: SBIA Group <sbia-software at uphs.upenn.edu>
 */

#ifndef __itkJointSegmentationRegistrationFunction_txx
#define __itkJointSegmentationRegistrationFunction_txx

#include <itkExceptionObject.h>
#include <vnl/vnl_math.h>
#include <vnl/algo/vnl_qr.h>
#include <itkNiftiImageIO.h>
//
#include "itkJointSegmentationRegistrationFunction.h"
#include "MyUtils.h"
#include "Volume.h"
//
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
#include "fams.h"
#endif


#define PI						3.1415926535897932384626433832795
#define PIM2					6.283185307179586476925286766559
#define PI2						9.8696044010893586188344909998762
#define PI2M4					39.478417604357434475337963999505
#define eps						1e-8
#define epss					1e-32
#define INITIAL_MEAN_VALUE		1e32


namespace itk {


// Construction/Destruction
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
JointSegmentationRegistrationFunction<TFixedImage, TMovingImage, TDeformationField, VNumberOfFixedChannels, VNumberOfMovingChannels>
::JointSegmentationRegistrationFunction()
{
	RadiusType r;
	unsigned int j;
	for (j = 0; j < ImageDimension; j++) {
		r[j] = 0;
	}
	this->SetRadius(r);

	m_TimeStep = 1.0;
	// this is a conservative choice //
	SetSigma2(1.0);
	SetDeltaSigma2(0.0);
#ifdef USE_PROB_PRIOR
	SetProbWeight(0.0);
#endif

	std::cout << "creating moving and fixed image vector objects .." << std::endl;
	m_FixedImageVector.resize(NumberOfFixedChannels, NULL);
	m_MovingImageVector.resize(NumberOfMovingChannels, NULL);
	m_MovingImageInterpolatorVector.resize(NumberOfMovingChannels, NULL);
	m_MovingImageWarperVector.resize(NumberOfMovingChannels, NULL);
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	std::cout << "creating weight vector objects .." << std::endl;
	m_WeightImageVector.resize(NumberOfMovingChannels, NULL);
#else
	std::cout << "creating weight image vector objects .." << std::endl;
	m_WeightImageVector.resize(NumberOfMovingChannels, NULL);
	for (int i = 1; i <= NumberOfMovingChannels; i++) {
		m_WeightImageVector[i-1] = WeightImageType::New();
	}
	std::cout << "populating the weight image vector done." << std::endl; 
#endif
#ifdef USE_PROB_PRIOR
	std::cout << "creating prob vector objects .." << std::endl;
	m_ProbImageVector.resize(NumberOfMovingChannels, NULL);
#endif

	// initializing mean and variance vectors
	for (int i = 1; i <= NumberOfMovingChannels; i++) {
		MeanType v(INITIAL_MEAN_VALUE);
		GetMeanVector()->push_back(v); 
		GetInitMeanVector()->push_back(v); 

		VarianceType s;
		s.set_identity();
		GetVarianceVector()->push_back(s);
	}

	std::cout << "populating the means and covarince vectors done." << std::endl; 
	// we dont have information on input images at this point so:
	m_FixedImageSpacing.Fill(1.0);
	m_FixedImageOrigin.Fill(0.0);
	m_FixedImageDirection.SetIdentity();
	m_Normalizer = 0.0;

	m_Metric = NumericTraits<double>::max();
	m_RMSChange = NumericTraits<double>::max();
	m_NumberOfPixelsProcessed = 0;
	m_SumOfEMLogs = 0;
	m_SumOfSquaredChange = 0;

	m_bUpdateMeansAndVariances = true;
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
	m_bMeanShiftUpdate = false;
	//
	mean_shift_update = false;
	ms_tumor = false;
	ms_edema = false;
#endif
}

// Debugging
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void JointSegmentationRegistrationFunction<TFixedImage, TMovingImage, TDeformationField, VNumberOfFixedChannels, VNumberOfMovingChannels>
::LogVariables()
{
	std::ofstream fout;                // output file
	fout.open("variables_log.txt", std::ios::out | std::ios::app);
	if (this->GetNumberOfElapsedIterations() == 0) {
		fout << "iteration:" << this->GetNumberOfElapsedIterations() << ", metric:" << GetMetric() << std::endl;
	} else {
		fout << "iteration:" << this->GetNumberOfElapsedIterations() << ", metric:" << GetMetric() << ", rms_change:" << GetRMSChange() << ", pixel_processed:" << m_NumberOfPixelsProcessed << std::endl;
	}
	fout.close();

	fout.open("cost.txt");
	fout << GetMetric() << std::endl;
	fout.close();
}

// Computes the posteriors  
static
inline void ComputeLikelihood4(double m[16], double ym[4], double* like) {
	double inv[16], det, det_1, ySIy;
	
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
	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	if (det < eps) {
		*like = 0;
		return;
	} else {
		det_1 = 1.0 / det;
	}
#else
	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12] + epss;
	det_1 = 1.0 / det;
#endif

	ySIy = ym[0] * (inv[ 0] * ym[0] + inv[ 1] * ym[1] + inv[ 2] * ym[2] + inv[ 3] * ym[3]) +
	       ym[1] * (inv[ 4] * ym[0] + inv[ 5] * ym[1] + inv[ 6] * ym[2] + inv[ 7] * ym[3]) +
	       ym[2] * (inv[ 8] * ym[0] + inv[ 9] * ym[1] + inv[10] * ym[2] + inv[11] * ym[3]) +
	       ym[3] * (inv[12] * ym[0] + inv[13] * ym[1] + inv[14] * ym[2] + inv[15] * ym[3]);
	ySIy *= det_1;

	*like = 1.0 / (PI2M4 * vcl_sqrt(det)) * exp (-0.5 * ySIy);
}
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
static
inline void ComputeLikelihood1(double m, double ym, double* like) {
	//*	
	double det;

#ifndef USE_SUM_EPSS
	det = m * m * m * m;
	if (det < eps) {
		*like = 0;
		return;
	}
#else
	det = m * m * m * m + epss;
#endif

	*like = 1.0 / (PI2M4 * vcl_sqrt(det)) * exp (-0.5 * ym * ym / m * 4);
	/*/
	if (m < eps) {
		*like = 0;
		return;
	}

	*like = 1.0 / (PI2M4 * vcl_sqrt(det)) * exp (-0.5 * ym * ym / m);
	//*/
}

static
inline void SolveLinearSystem3(double m[9], double b[3], double* x) {
	double inv[9], det, det_1;

	inv[0] =  (m[4] * m[8] - m[7] * m[5]);
	inv[1] = -(m[3] * m[8] - m[5] * m[6]);
	inv[2] =  (m[3] * m[7] - m[4] * m[6]);
	inv[3] = -(m[1] * m[8] - m[2] * m[7]);
	inv[4] =  (m[0] * m[8] - m[2] * m[6]);
	inv[5] = -(m[0] * m[7] - m[1] * m[6]);
	inv[6] =  (m[1] * m[5] - m[2] * m[4]);
	inv[7] = -(m[0] * m[5] - m[2] * m[3]);
	inv[8] =  (m[0] * m[4] - m[1] * m[3]);

#ifndef USE_SUM_EPSS
	det = m[0] * inv[0] - m[1] * inv[1] + m[2] * inv[2];
	if (det < eps) {
		x[0] = x[1] = x[2] = 0;
		return;
	} else {
		det_1 = 1.0 / det;
	}
#else
	det = m[0] * inv[0] - m[1] * inv[1] + m[2] * inv[2] + epss;
	det_1 = 1.0 / det;
#endif

	x[0] = inv[0] * b[0] + inv[1] * b[1] + inv[2] * b[2];
	x[1] = inv[3] * b[0] + inv[4] * b[1] + inv[5] * b[2];
	x[2] = inv[6] * b[0] + inv[7] * b[1] + inv[8] * b[2];

	x[0] *= det_1;
	x[1] *= det_1;
	x[2] *= det_1;
}

#ifdef USE_4_IMAGES
#define USE_FAST_LIKELIHOOD
#endif

template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void JointSegmentationRegistrationFunction<TFixedImage, TMovingImage, TDeformationField, VNumberOfFixedChannels, VNumberOfMovingChannels>
::ComputeWeightImages(bool bUpdateWeightImages)
{
	int i, j, k;
    double cost = 0.0;
    double number_of_pixels = 0.0;
#ifdef USE_PROB_PRIOR
	double cost_prior = 0.0;
#endif
#ifdef USE_FAST_LIKELIHOOD
	double vv[NumberOfMovingChannels][16];
	double vv_inv[NumberOfMovingChannels][16];
	double vv_det[NumberOfMovingChannels], vv_c1[NumberOfMovingChannels], vv_c2[NumberOfMovingChannels];
	double mv[NumberOfMovingChannels][4];

	// assume 4 channels for image
	if (NumberOfFixedChannels != 4) {
		std::cout << "NumberOfFixedChannels must be 4 to use fast likelihood" << std::endl;
		return;
	}

	for (i = 0; i < NumberOfMovingChannels; i++) {
		for (j = 0; j < 4; j++) {
			for (k = 0; k < 4; k++) {
				vv[i][j*4+k] = GetVarianceVector()->at(i)(j, k);
			}
			mv[i][j] = GetMeanVector()->at(i)(j);
		}
	}
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

	// define iterators over moving images (warped priors)
	std::vector<MovingImageConstIteratorType> m_it_vect;
	for (i = 1; i <= NumberOfMovingChannels; i++) {
		MovingImageConstIteratorType m_it(GetNthImageWarper(i)->GetOutput(), GetNthImageWarper(i)->GetOutput()->GetLargestPossibleRegion());	
		m_it.GoToBegin();
		m_it_vect.push_back(m_it);
	}
	// define iterators over fixed images
	std::vector<FixedImageConstIteratorType> f_it_vect;
	for (i = 1; i <= NumberOfFixedChannels; i++) {
		FixedImageConstIteratorType f_it(GetNthFixedImage(i), GetNthFixedImage(i)->GetLargestPossibleRegion());
		f_it.GoToBegin();
		f_it_vect.push_back(f_it);
	}
	// define iterators over weight images
	std::vector<WeightImageIteratorType> w_it_vect;
	for (i = 1; i <= NumberOfMovingChannels; i++) {
		WeightImageIteratorType w_it(GetNthWeightImage(i), GetNthWeightImage(i)->GetLargestPossibleRegion());
		w_it.GoToBegin();
		w_it_vect.push_back(w_it);
	}
#ifdef USE_PROB_PRIOR
	// define iterators over prob images
	std::vector<ProbImageConstIteratorType> p_it_vect;
	for (i = 1; i <= NumberOfMovingChannels; i++) {
		ProbImageConstIteratorType p_it(GetNthProbImage(i), GetNthProbImage(i)->GetLargestPossibleRegion());	
		p_it.GoToBegin();
		p_it_vect.push_back(p_it);
	}
#endif

	// looooping through all voxels
	while (!w_it_vect[0].IsAtEnd()) {
		double prior[NumberOfMovingChannels];
		double like[NumberOfMovingChannels];
		double post[NumberOfMovingChannels];
#ifdef USE_PROB_PRIOR
		double prob[NumberOfMovingChannels];
#endif
#ifdef USE_FAST_LIKELIHOOD
		double y[4], ym[4];
#else
		MeanType y;
#endif
#ifndef USE_DISCARD_ZERO_AREA
		double ys = 0;
#else
		double ys = 1;
#endif

		number_of_pixels++;

		// making a vnl vector from fixed images
		for (i = 0; i < NumberOfFixedChannels; i++) {
#ifdef USE_FAST_LIKELIHOOD
			y[i] = f_it_vect[i].Get();
#ifndef USE_DISCARD_ZERO_AREA
			ys += y[i];
#else
			if (y[i] <= 0) {
				ys = 0;
			}
#endif
#else
			y(i) = f_it_vect[i].Get();
#ifndef USE_DISCARD_ZERO_AREA
			ys += y(i);
#else
			if (y(i) <= 0) {
				ys = 0;
			}
#endif
#endif
		}

		// read priors
		for (i = 0; i < NumberOfMovingChannels; i++) {
			prior[i] = m_it_vect[i].Get();
#if defined(USE_PROB_PRIOR) && !defined(PROB_LOAD_TU_ONLY)
			prob[i] = p_it_vect[i].Get(); 
#endif
		}
#if defined(USE_PROB_PRIOR) && defined(PROB_LOAD_TU_ONLY)
		prob[TU] = p_it_vect[TU].Get(); 
#endif

		// computing likelihood values
#ifndef USE_COST_BG
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
		if (ys > 0 || (prior[TU]+prior[NCR]+prior[NE]) > 0.01) {
#else
		//if (ys > 0) {
		if (ys > 0 || (prior[TU]+prior[NCR]) > 0.01) {
#endif
#else
		if (1) {
#endif
			// to the number of classes
			for (i = 0; i < NumberOfMovingChannels; i++) {
#ifdef USE_FAST_LIKELIHOOD
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
#else
				MeanType SIy = vnl_qr<double>(GetVarianceVector()->at(i)).solve(y - GetMeanVector()->at(i));
				double ySIy = dot_product(y - GetMeanVector()->at(i), SIy);
				double detS = vnl_determinant(GetVarianceVector()->at(i)) + eps;
				like[i] =  1.0 / (PI2M4 * vcl_sqrt(detS)) * exp(-.5*ySIy);
#endif
			}
			if (bUpdateWeightImages) {
#ifndef USE_LIKE_TU_MAX
				// compute the denum
#ifndef USE_SUM_EPSS2
				double denum = 0;
#else
				double denum = epss;
#endif
				for (i = 0; i < NumberOfMovingChannels; i++) {
					denum += prior[i] * like[i];
				}
#ifndef USE_SUM_EPSS2
				if (denum < eps) {
					denum = eps;
				}
#else
#endif
				// compute the posterior
				for (i = 0; i < NumberOfMovingChannels; i++) {
					post[i] = prior[i] * like[i] / denum;
				}
#else
				// compute the denum
#ifndef USE_SUM_EPSS2
				double denum = 0.0;
#else
				double denum = epss;
#endif
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
				for (i = 0; i < NumberOfMovingChannels; i++) {
					post[i] = w_it_vect[i].Get();
				}
			}
		} else {
			// compute the posterior
			for (i = 0; i < NumberOfMovingChannels; i++) {
				post[i] = 0.0;
				like[i] = 1.0;
			}
#ifdef USE_WARPED_BG
#ifndef USE_JSR_NO_WARPED_BG
			post[BG] = 1.0;
#endif
#endif
		}
		//if (bUpdateWeightImages) {
			for (i = 0; i < NumberOfMovingChannels; i++) {
				w_it_vect[i].Set(post[i]);
			}
		//}

		for (i = 0; i < NumberOfMovingChannels; i++) {
			// we have to take care of negative values
			double val = like[i] * prior[i];
#ifndef USE_SUM_EPSS2
			if (val < epss) val = epss;
#else
			if (val <= 0.0) val = epss;
#endif
			cost -= post[i] * log(val);
		}
#ifdef USE_PROB_PRIOR
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
		cost_prior += m_ProbWeight * 9 * (prior[TU] - prob[TU]) * (prior[TU] - prob[TU]);
#else
		// multiplying 4 is considering tu_prior = prior[TU] + prior[NCR]
		cost_prior += m_ProbWeight * 4 * (prior[TU] - prob[TU]) * (prior[TU] - prob[TU]);
		//cost_prior += m_ProbWeight * (post[TU] + post[NCR]) * ((prior[TU] - prob[TU]) * (prior[TU] - prob[TU]));
#endif
#endif

		// forwarding the iterators
		for (i = 0; i < NumberOfFixedChannels; i++) {
			++f_it_vect[i];
		}
		for (i = 0; i < NumberOfMovingChannels; i++) {
			++w_it_vect[i];
			++m_it_vect[i];
#if defined(USE_PROB_PRIOR) && !defined(PROB_LOAD_TU_ONLY)
			++p_it_vect[i];
#endif
		}
#if defined(USE_PROB_PRIOR) && defined(PROB_LOAD_TU_ONLY)
		++p_it_vect[TU];
#endif
	}

#if 1
	m_SumOfEMLogs = cost;
#ifdef USE_PROB_PRIOR
	m_SumOfEMLogs += cost_prior;
#endif
	m_Metric = m_SumOfEMLogs / number_of_pixels; 
#endif

	cost /= number_of_pixels;
#ifdef USE_PROB_PRIOR
	cost_prior /= number_of_pixels;
#endif

#ifdef USE_PROB_PRIOR
	std::cout << "cost = " << cost << ", cost_prior = " << cost_prior << ", sum = " << cost + cost_prior << ", number_of_pixels = " << (int)number_of_pixels << std::endl;
#else
	std::cout << "cost = " << cost << ", number_of_pixels = " << (int)number_of_pixels << std::endl;
#endif

	// releasing memory 
	for (i = 0; i < NumberOfFixedChannels; i++) {
		f_it_vect.pop_back();
	}
	for (i = 0; i < NumberOfMovingChannels; i++) {
		w_it_vect.pop_back();
		m_it_vect.pop_back();
#ifdef USE_PROB_PRIOR
		p_it_vect.pop_back();
#endif
	}
}

#define MAX_LEVELS 300
template <class A>
void quickSort(A *arr, int elements)
{
	int beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R, swap;
	A piv;
	beg[0]=0; end[0]=elements;
	while (i>=0) {
		L=beg[i]; R=end[i]-1;
		if (L<R) {
			piv=arr[L];
			while (L<R) {
				while (arr[R]>=piv && L<R) R--; 
				if (L<R) arr[L++]=arr[R];
				while (arr[L]<=piv && L<R) L++; 
				if (L<R) arr[R--]=arr[L]; 
			}
			arr[L]=piv; beg[i+1]=L+1; end[i+1]=end[i]; end[i++]=L;
			if (end[i]-beg[i]>end[i-1]-beg[i-1]) {
				swap=beg[i]; beg[i]=beg[i-1]; beg[i-1]=swap;
				swap=end[i]; end[i]=end[i-1]; end[i-1]=swap; 
			}
		} else {
			i--; 
		}
	}
}

#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
//#define USE_MS_TRACE
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
BOOL
JointSegmentationRegistrationFunction<TFixedImage, TMovingImage, TDeformationField, VNumberOfFixedChannels, VNumberOfMovingChannels>
::ComputeMSMap(std::vector<FixedImageConstIteratorType>& f_it_vect, BVolume& label_map, FVolume* ms_map, int* rl, int rl_num, int* cl, int cl_num, int samp) 
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
	double mv_init[NumberOfMovingChannels][NumberOfFixedChannels];
	int l, m, n, i, j, k;
	int **rl_merge, *rl_merge_num;
	//
	BOOL bRes = TRUE;

	vd_x = label_map.m_vd_x;
	vd_y = label_map.m_vd_y;
	vd_z = label_map.m_vd_z;

	for (k = 0; k < NumberOfMovingChannels; k++) {
		for (i = 0; i < NumberOfFixedChannels; i++) {
			mv_init[k][i] = GetInitMeanVector()->at(k)[i];
			//mv_init[k][i] = GetMeanVector()->at(k)[i];
		}
	}

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
    int n_ = 0, d_ = NumberOfFixedChannels;

		//
		samp_test.allocate(vd_x, vd_y, vd_z);
		pttemp = new float[vd_x * vd_y * vd_z * d_];

		for (n = 0; n < vd_z; n++) 
    {
			for (m = 0; m < vd_y; m++) 
      {
				for (l = 0; l < vd_x; l++) 
        {
					samp_test.m_pData[n][m][l][0] = 0;
          if ((n % samp == 0) && (m % samp == 0) && (l % samp == 0))
          {
            samp_test.m_pData[n][m][l][0] = 1;
            int val = label_map.m_pData[n][m][l][0];
            BOOL bRgn = FALSE;
            for (i = 0; i < rl_num; i++) 
            {
              if (val == label_idx[rl[i]]) 
              {
                bRgn = TRUE;
                break;
              }
            }
            if (bRgn) 
            {
              for (i = 0; i < d_; i++) 
              {
                pttemp[n_*d_ + i] = f_it_vect[i].Get();
              }
              n_++;
            }
          }

          for (i = 0; i < NumberOfFixedChannels; i++) 
          {
            ++f_it_vect[i];
          }
				}
			}
		}

		for (i = 0; i < NumberOfFixedChannels; i++) 
    {
			f_it_vect[i].GoToBegin();
		}

#ifdef USE_MS_TRACE
		TRACE2("n_ = %d, d_ = %d\n", n_, d_);
#endif
		if (n_ < 10) {
			TRACE("n_ < 10: quit\n");
			bRes = FALSE;
		} else {
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
		}

		delete [] pttemp;
	}

	if (bRes) {
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
#ifdef USE_MS_TRACE
			TRACE2("Found K = %d L = %d\n", K, L);
#endif
		}
#else
		K = 30;
		L = 6;
#ifdef USE_MS_TRACE
		TRACE2("Using K = %d L = %d\n", K, L);
#endif
#endif
#ifdef USE_MS_TRACE
		TRACE2("k_neigh = %d\n", k_neigh);
#endif

		//sprintf(fdata_file_name, "%s%sfams_pilot_%d.txt", tmp_folder, DIR_SEP, k_neigh);
		fams->RunFAMS(K, L, k_neigh, percent, jump, width, fdata_file_name);
	
#ifdef USE_MS_TRACE
		TRACE2("fams->npm_ = %d\n", fams->npm_);
#endif
		if (fams->npm_ < rl_num) {
			TRACE("fams->npm_ < rl_num: quit\n");
			bRes = FALSE;
		} else {
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
#ifdef USE_MS_TRACE
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
#ifdef USE_MS_TRACE
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
				for (n = 0; n < vd_z; n++) 
        {
					for (m = 0; m < vd_y; m++) 
          {
						for (l = 0; l < vd_x; l++) 
            {
							for (i = 0; i < rl_num; i++) 
              {
								ms_map[rl[i]].m_pData[n][m][l][0] = 0;
							}
              if ((n % samp == 0) && (m % samp == 0) && (l % samp == 0))
              {
                int val = label_map.m_pData[n][m][l][0];
                BOOL bRgn = FALSE;
                for (i = 0; i < rl_num; i++) 
                {
                  if (val == label_idx[rl[i]]) 
                  {
                    bRgn = TRUE;
                    break;
                  }
                }
                if (bRgn) 
                {
                  for (i = 0; i < fams->npm_; i++) 
                  {
                    if (fams->modes_[idx] == fams->prunedmodes_[i*fams->d_] &&
                      fams->modes_[idx + 1] == fams->prunedmodes_[i*fams->d_ + 1] &&
                      fams->modes_[idx + 2] == fams->prunedmodes_[i*fams->d_ + 2] &&
                      fams->modes_[idx + 3] == fams->prunedmodes_[i*fams->d_ + 3]) 
                    {
                      int pl = prunedlabel[i];
                      if (pl < 0) 
                      {
                        // label matched to outliers
                      }
                      else 
                      {
                        ms_map[rl[pl]].m_pData[n][m][l][0] = 1;
                        for (k = 0; k < rl_merge_num[pl]; k++) 
                        {
                          ms_map[rl[rl_merge[pl][k]]].m_pData[n][m][l][0] = 1;
                        }
                        for (k = 0; k < rl_num; k++) 
                        {
                          if (nnmode[k] == i) 
                          {
                            ms_map[rl[k]].m_pData[n][m][l][0] = 1;
                          }
                        }
                      }
                      break;
                    }
                  }
                  if (i == fams->npm_) 
                  {
                    // error
                    std::cerr << "(i == fams->npm_) in itkJoinSegmentationRegistrationFunction.\n";
                  }
                  //
                  idx += fams->d_;
                }
              }
						}
					}
				}
			}

			MyFree(prunedlabel);
			MyFree(nnmode);
			MyFree(cl_dist);
			MyFree(rl_dist);
		}
	}

	for (i = 0; i < rl_num; i++) {
		MyFree(rl_merge[i]);
	}
	MyFree(rl_merge);
	MyFree(rl_merge_num);

	delete fams;

	return bRes;
}
#endif

// Updates the means and variances  
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
JointSegmentationRegistrationFunction<TFixedImage, TMovingImage, TDeformationField, VNumberOfFixedChannels, VNumberOfMovingChannels>
::UpdateMeansAndVariances(void)
{
	// define a vector object to keep sum of weighted vectors for each class
	double mv_sum[NumberOfMovingChannels][NumberOfFixedChannels];
	// define a vector object to keep sum of weighted outer products for each class
	double vv_sum[NumberOfMovingChannels][NumberOfFixedChannels*NumberOfFixedChannels];
	// define a vector object to keep sum of the weights for each class
	double w_sum[NumberOfMovingChannels];
	double mv[NumberOfMovingChannels][NumberOfFixedChannels];
	int i, j, k;
	//
#if defined(USE_OUTLIER_REJECTION_INIT_MEANS) || defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
	int l, m, n;
	const SizeType size = GetNthFixedImage(1)->GetLargestPossibleRegion().GetSize();
	int vd_x = size[0];
	int vd_y = size[1];
	int vd_z = size[2];
#endif
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
				p_vv[i][j*4+k] = GetVarianceVector()->at(i)(j, k);
			}
			p_mv[i][j] = GetMeanVector()->at(i)(j);
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
		if (this->GetNumberOfElapsedIterations()) {
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
	if (this->GetNumberOfElapsedIterations()) {
		use_like = true;
	}
#endif

	// define some iterators over fixed images
	std::vector<FixedImageConstIteratorType> f_it_vect;
	for (i = 1; i <= NumberOfFixedChannels; i++) {
		FixedImageConstIteratorType f_it(GetNthFixedImage(i), GetNthFixedImage(i)->GetLargestPossibleRegion());
		f_it.GoToBegin();
		f_it_vect.push_back(f_it);
	}

	// define some iterators over weight images
	std::vector<WeightImageIteratorType> w_it_vect;
	for (i = 1; i <= NumberOfMovingChannels; i++) {
		WeightImageIteratorType w_it(GetNthWeightImage(i), GetNthWeightImage(i)->GetLargestPossibleRegion());
		w_it.GoToBegin();
		w_it_vect.push_back(w_it);
	}

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
				mv_init[k][i] = GetInitMeanVector()->at(k)[i];
			}
		}

		for (i = 0; i < NumberOfMovingChannels; i++) {
			dist_hist_n[i] = 0;
			dist_hist[i] = (int*)malloc((hist_max+1) * sizeof(int));
			for (j = 0; j <= hist_max; j++) {
				dist_hist[i][j] = 0;
			}
		}

		while (!w_it_vect.at(0).IsAtEnd()) {
			double y[NumberOfFixedChannels];
			const IndexType index = w_it_vect.at(0).GetIndex();
			l = index[0];
			m = index[1];
			n = index[2];

			for (i = 0; i < NumberOfMovingChannels; i++) {
				dist_map[i].m_pData[n][m][l][0] = hist_max;
			}

#ifndef USE_DISCARD_ZERO_AREA
			double ys = 0;
#else
			double ys = 1;
#endif
			//
			for (i = 0; i < NumberOfFixedChannels; i++) {
				y[i] = f_it_vect[i].Get();
#ifndef USE_DISCARD_ZERO_AREA
				ys += y[i];
#else
				if (y[i] <= 0) {
					ys = 0;
				}
#endif
			}
			if (ys > 0) {
				for (i = 0; i < NumberOfMovingChannels; i++) {
					if ((double)w_it_vect[i].Get() > p_th) {
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

			for (i = 0; i < NumberOfFixedChannels; i++) {
				++f_it_vect[i];
			}
			for (i = 0; i < NumberOfMovingChannels; i++) {
				++w_it_vect[i];
			}
		}
		for (i = 0; i < NumberOfFixedChannels; i++) {
			f_it_vect[i].GoToBegin();
		}
		for (i = 0; i < NumberOfMovingChannels; i++) {
			w_it_vect[i].GoToBegin();
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
							if ((double)w_it_vect.at(ED).Get() > 0) {
								dist_map[i].m_pData[n][m][l][0] = 0;
							} else {
								dist_map[i].m_pData[n][m][l][0] = 1;
							}
#endif
						} else {
							dist_map[i].m_pData[n][m][l][0] = 1;
						}
					}
					for (i = 0; i < NumberOfMovingChannels; i++) {
						++w_it_vect[i];
					}
				}
			}
		}
		for (i = 0; i < NumberOfMovingChannels; i++) {
			w_it_vect[i].GoToBegin();
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
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
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

	if (mean_shift_update && m_bMeanShiftUpdate) {
		BVolume label_map;
		long long size_tu = 0;
		long long size_wm_ed = 0;

		label_map.allocate(vd_x, vd_y, vd_z);

		while (!w_it_vect[0].IsAtEnd()) {
			const IndexType index = w_it_vect[0].GetIndex();
			l = index[0];
			m = index[1];
			n = index[2];

			float max_l_val = w_it_vect[0].Get();
			int max_l_idx = 0;
			for (i = 1; i < NumberOfMovingChannels; i++) {
				if (max_l_val < w_it_vect[i].Get()) {
					max_l_val = w_it_vect[i].Get();
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

			for (i = 0; i < NumberOfMovingChannels; i++) {
				++w_it_vect[i];
			}
		}
		for (i = 0; i < NumberOfMovingChannels; i++) {
			w_it_vect[i].GoToBegin();
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
				cl_num = 2;
				//cl[2] = ED;
				//cl_num = 3;
#endif
				TRACE2("size_tu = %d\n", size_tu);
				if (size_tu > 800000) {
					TRACE2("size_tu > 800000: skip ComputeMSMap\n");
				} else if (size_tu > 100000) {
					ComputeMSMap(f_it_vect, label_map, ms_map, rl, rl_num, cl, cl_num, 2);
				} else {
					ComputeMSMap(f_it_vect, label_map, ms_map, rl, rl_num, cl, cl_num, 1);
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
					ComputeMSMap(f_it_vect, label_map, ms_map, rl, rl_num, cl, cl_num, 2);
				} else { 
					ComputeMSMap(f_it_vect, label_map, ms_map, rl, rl_num, cl, cl_num, 1);
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
				ComputeMSMap(f_it_vect, label_map, ms_map, rl, rl_num, cl, cl_num, 2);
				//*/
				//
				/*
				rl[0] = GM;
				rl_num = 1;
				cl[0] = GM;
				cl[1] = WM;
				cl[2] = ED;
				cl_num = 3;
				ComputeMSMap(f_it_vect, label_map, ms_map, rl, rl_num, cl, cl_num, 2);
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
				ComputeMSMap(f_it_vect, label_map, ms_map, rl, rl_num, cl, cl_num);
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
#endif

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
	while (!w_it_vect[0].IsAtEnd()) {
		double y[NumberOfFixedChannels];
#ifdef USE_OUTLIER_REJECTION_LAMBDA
		double p_ym[NumberOfFixedChannels];
		double p_like[NumberOfMovingChannels];
#endif
#ifndef USE_DISCARD_ZERO_AREA
		double ys = 0;
#else
		double ys = 1;
#endif
		//
		for (i = 0; i < NumberOfFixedChannels; i++) {
			y[i] = f_it_vect[i].Get();
#ifndef USE_DISCARD_ZERO_AREA
			ys += y[i];
#else
			if (y[i] <= 0) {
				ys = 0;
			}
#endif
		}
		if (ys > 0) {
#if defined(USE_OUTLIER_REJECTION_INIT_MEANS) || defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
			const IndexType index = w_it_vect[0].GetIndex();
			l = index[0];
			m = index[1];
			n = index[2];
#endif
			//
			for (k = 0; k < NumberOfMovingChannels; k++) {
				double p;
				//
#ifdef USE_OUTLIER_REJECTION_INIT_MEANS
				if (dist_map[k].m_pData[n][m][l][0] == 0) {
					continue;
				}
#endif
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
				if (ms_map[k].m_pData[n][m][l][0] == 0) {
					continue;
				}
#endif
				//
				p = (double)w_it_vect[k].Get();
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
		// now forwarding all iterators
		for (i = 0; i < NumberOfFixedChannels; i++) {
			++f_it_vect[i];
		}
		for (i = 0; i < NumberOfMovingChannels; i++) {
			++w_it_vect[i];
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
	if (this->GetNumberOfElapsedIterations()) {
		for (k = 0; k < NumberOfMovingChannels; k++) {
			double w_1 = 1.0 / (w_sum[k] + eps);
			for (i = 0; i < NumberOfFixedChannels; i++) {
				mv[k][i] = w_1 * mv_sum[k][i] + eps;
				GetMeanVector()->at(k)[i] = mv[k][i];
			}
		}
	} else {
		// in the first iteration we want to favor user supplied values
		for (k = 0; k < NumberOfMovingChannels; k++) {
			double w_1 = 1.0 / (w_sum[k] + eps);
			for (i = 0; i < NumberOfFixedChannels; i++) {
				mv[k][i] = w_1 * mv_sum[k][i] + eps;
				if (GetMeanVector()->at(k)[i] == INITIAL_MEAN_VALUE) {
					GetMeanVector()->at(k)[i] = mv[k][i];
				} else {
					mv[k][i] = GetMeanVector()->at(k)[i];
				}
			}
		}
	}
	
	// now compute the covariances
	// first sending iterator to begining
	for (i = 0; i < NumberOfFixedChannels; i++) {
		f_it_vect[i].GoToBegin();
	}
	for (i = 0; i < NumberOfMovingChannels; i++) {
		w_it_vect[i].GoToBegin();
	}

	// sum loop for variances
	while (!w_it_vect[0].IsAtEnd()) {
		double y[NumberOfFixedChannels];
		double ym[NumberOfFixedChannels];
#ifdef USE_OUTLIER_REJECTION_LAMBDA
		double p_ym[NumberOfFixedChannels];
		double p_like[NumberOfMovingChannels];
#endif
#ifndef USE_DISCARD_ZERO_AREA
		double ys = 0;
#else
		double ys = 1;
#endif
		//
		for (i = 0; i < NumberOfFixedChannels; i++) {
			y[i] = f_it_vect[i].Get();
#ifndef USE_DISCARD_ZERO_AREA
			ys += y[i];
#else
			if (y[i] <= 0) {
				ys = 0;
			}
#endif
		}
		if (ys > 0) {
#if defined(USE_OUTLIER_REJECTION_INIT_MEANS) || defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
			const IndexType index = w_it_vect[0].GetIndex();
			l = index[0];
			m = index[1];
			n = index[2];
#endif
			//
			for (k = 0; k < NumberOfMovingChannels; k++) {
				double p;
				//
#ifdef USE_OUTLIER_REJECTION_INIT_MEANS
				if (dist_map[k].m_pData[n][m][l][0] == 0) {
					continue;
				}
#endif
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
				if (ms_map[k].m_pData[n][m][l][0] == 0) {
					continue;
				}
#endif
				//
				p = (double)w_it_vect[k].Get();
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
		// now forwarding all iterators
		for (i = 0; i < NumberOfFixedChannels; i++) {
			++f_it_vect[i];
		}
		for (i = 0; i < NumberOfMovingChannels; i++) {
			++w_it_vect[i];
		}
	}

	// computing updated variances to the number of classes
	for (k = 0; k < NumberOfMovingChannels; k++) {
		double w_1 = 1.0 / (w_sum[k] + eps);
		for (j = 0; j < NumberOfFixedChannels; j++) {
			for (i = 0; i < NumberOfFixedChannels; i++) {
				GetVarianceVector()->at(k)[j][i] = w_1 * vv_sum[k][NumberOfFixedChannels*j+i];
			}
		}
	}

#ifdef USE_OUTLIER_REJECTION_INIT_MEANS
	for (i = 0; i < NumberOfMovingChannels; i++) {
		dist_map[i].clear();
	}
#endif
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
	for (i = 0; i < NumberOfMovingChannels; i++) {
		ms_map[i].clear();
	}
#endif
}

// Standard "PrintSelf" method.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
JointSegmentationRegistrationFunction<TFixedImage, TMovingImage, TDeformationField, VNumberOfFixedChannels, VNumberOfMovingChannels>
::PrintSelf(std::ostream& os, Indent indent) const
{
	os << indent << "Not implemented yet!" << std::endl;
}

// Preparing the function before each iteration
#if defined(USE_8_PRIORS)
enum {
	BG = 0,
    CSF,
    GM,
    WM,
    VS,
    ED,
    NCR,
    TU
};
static const char label[8][32] = { "BG", "CSF", "GM", "WM", "VS", "ED", "NCR", "TU" };
static const int labeln[8] = { 7, 6, 4, 1, 2, 3, 5, 0 };
#elif defined(USE_9_PRIORS)
enum {
	BG = 0,
    CSF,
    VT,
    GM,
    WM,
    VS,
    ED,
    NCR,
    TU
};
static const char label[9][32] = { "BG", "CSF", "VT", "GM", "WM", "VS", "ED", "NCR", "TU" };
#elif defined(USE_10_PRIORS)
enum {
	BG = 0,
    CSF,
    VT,
    GM,
    WM,
    VS,
    ED,
    NCR,
    TU,
	NE
};
static const char label[10][32] = { "BG", "CSF", "VT", "GM", "WM", "VS", "ED", "NCR", "TU", "NE" };
#elif defined(USE_10A_PRIORS)
enum {
	BG = 0,
    CSF,
    GM,
    WM,
    VS,
    ED,
    NCR,
    TU,
	NE,
	CB
};
static const char label[10][32] = { "BG", "CSF", "GM", "WM", "VS", "ED", "NCR", "TU", "NE", "CB" };
#elif defined(USE_11_PRIORS)
enum {
	BG = 0,
    CSF,
    VT,
    GM,
    WM,
    VS,
    ED,
    NCR,
    TU,
	NE,
	CB
};
static const char label[11][32] = { "BG", "CSF", "VT", "GM", "WM", "VS", "ED", "NCR", "TU", "NE", "CB" };
#elif defined(USE_11A_PRIORS)
enum {
	BG = 0,
    CSF,
    VT,
    GM,
    WM,
    VS,
    ED,
    NCR,
    TU,
	RTN,
	RTE
};
static const char label[11][32] = { "BG", "CSF", "VT", "GM", "WM", "VS", "ED", "NCR", "TU", "RTN", "RTE" };
#endif

template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
BOOL
JointSegmentationRegistrationFunction<TFixedImage, TMovingImage, TDeformationField, VNumberOfFixedChannels, VNumberOfMovingChannels>
::SaveMeansAndVariances(const char* means_file, const char* variances_file, MeanVectorType* pMeanVector, VarianceVectorType* pVarianceVector)
{
	FILE* fp;
	int i, j, k;

	////////////////////////////////////////////////////////////////////////////////
	// save means
	fp = fopen(means_file, "w");
	if (fp == NULL) {
		printf("Failed to open file: '%s'\n", means_file);
		return FALSE;
	}
	for(i = 0; i < VNumberOfMovingChannels; i++) {
		MeanType mean;
		//
#if defined(USE_8_PRIORS)
		mean = pMeanVector->at(labeln[i]);
		//
		fprintf(fp, "%s\n", label[labeln[i]]);
#else
		mean = pMeanVector->at(i);
		//
		fprintf(fp, "%s\n", label[i]);
#endif
		//
		for (j = 0; j < VNumberOfFixedChannels; j++) {
			fprintf(fp, "%f", mean(j));
			if (j == VNumberOfFixedChannels-1) {
				fprintf(fp, "\n");
			} else {
				fprintf(fp, " ");
			}
		}
	}
	fclose(fp);
	////////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////////
	// save variances
	fp = fopen(variances_file, "w");
	if (fp == NULL) {
		printf("Failed to open file: '%s'\n", variances_file);
		return FALSE;
	}
	for(i = 0; i < VNumberOfMovingChannels; i++) {
		VarianceType var;
		//
#if defined(USE_8_PRIORS)
		var = pVarianceVector->at(labeln[i]);
		//
		fprintf(fp, "%s\n", label[labeln[i]]);
#else
		var = pVarianceVector->at(i);
		//
		fprintf(fp, "%s\n", label[i]);
#endif
		//
		for (j = 0; j < VNumberOfFixedChannels; j++) {
			for (k = 0; k < VNumberOfFixedChannels; k++) {
				fprintf(fp, "%f", var(j, k));
				if (k == VNumberOfFixedChannels-1) {
					fprintf(fp, "\n");
				} else {
					fprintf(fp, " ");
				}
			}
		}
	}
	fclose(fp);
	////////////////////////////////////////////////////////////////////////////////

	return TRUE;
}

template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
JointSegmentationRegistrationFunction<TFixedImage, TMovingImage, TDeformationField, VNumberOfFixedChannels, VNumberOfMovingChannels>
::InitializeIteration()
{
	int i;

	std::cout << "function initialize iteration.." << std::endl;
	if (!this->GetNumberOfElapsedIterations()) {
		std::cout << "Cache fixed image information" << std::endl;

		m_FixedImageOrigin = GetNthFixedImage(1)->GetOrigin();
		m_FixedImageSpacing = GetNthFixedImage(1)->GetSpacing();
		m_FixedImageDirection = GetNthFixedImage(1)->GetDirection();

		// setting the dsigma2
		double dsigma2 = (1.0 - GetSigma2()) / (double)GetNumberOfMaximumIterations();
		SetDeltaSigma2(dsigma2);

		std::cout << "Initializing vector objects in function .." << std::endl;

		for (i = 1; i <= NumberOfMovingChannels; i++) {
			std::cout << "setting up moving image interpolators .." << std::endl;
			typename DefaultInterpolatorType::Pointer interp = DefaultInterpolatorType::New();
			SetNthImageInterpolator(static_cast<InterpolatorType*>(interp.GetPointer()), i);
			GetNthImageInterpolator(i)->SetInputImage(GetNthMovingImage(i));
			std::cout << "setting up moving image warpers .." << std::endl;
			SetNthImageWarper(WarperType::New(), i);
			GetNthImageWarper(i)->SetInterpolator(GetNthImageInterpolator(i));
			if ((i-1) == BG) {
				GetNthImageWarper(i)->SetEdgePaddingValue(1.0);
			} else {
				GetNthImageWarper(i)->SetEdgePaddingValue(0.0);
			}
			GetNthImageWarper(i)->SetOutputOrigin(this->m_FixedImageOrigin);
			GetNthImageWarper(i)->SetOutputSpacing(this->m_FixedImageSpacing);
			GetNthImageWarper(i)->SetOutputDirection(this->m_FixedImageDirection);
			GetNthImageWarper(i)->SetInput(GetNthMovingImage(i));
#if ITK_VERSION_MAJOR >= 4
			GetNthImageWarper(i)->SetDisplacementField(this->GetDisplacementField());
			GetNthImageWarper(i)->GetOutput()->SetRequestedRegion(this->GetDisplacementField()->GetRequestedRegion());
#else
			GetNthImageWarper(i)->SetDeformationField(this->GetDeformationField());
			GetNthImageWarper(i)->GetOutput()->SetRequestedRegion(this->GetDeformationField()->GetRequestedRegion());
#endif
		} // for
	} // if
	
	std::cout <<"warping moving images .."<< std::endl;

	for (i = 1; i <= NumberOfMovingChannels; i++) {
		GetNthImageWarper(i)->Update();
	}
	
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	//if (m_bUpdateMeansAndVariances && !this->GetNumberOfElapsedIterations()) {
	if (0) {
#else
	if (!this->GetNumberOfElapsedIterations()) {
#endif
		std::cout << "initialization of posteriors to the warped atlas" << std::endl;
		// setting weight images (posterior probs) into reasonable value
		for (i = 1; i <= NumberOfMovingChannels; i++) {
			WeightImageIteratorType w_it(GetNthWeightImage(i), GetNthWeightImage(i)->GetLargestPossibleRegion()); 
			w_it.GoToBegin();

			MovingImageConstIteratorType m_it(GetNthImageWarper(i)->GetOutput(), GetNthImageWarper(i)->GetOutput()->GetLargestPossibleRegion()); 
			m_it.GoToBegin();

			while (!w_it.IsAtEnd()) {
				if (m_it.Get() != NumericTraits<MovingPixelType>::max()) {
					w_it.Set(m_it.Get());
				} else {
					if ((i-1) == BG) {
						w_it.Set(1.0);
					} else {
						w_it.Set(0.0);
					}
				}
				++w_it;
				++m_it;
			}
		}
	}

	if (m_bUpdateMeansAndVariances) {
		std::cout << "updating means and variances ..." << std::endl;
		UpdateMeansAndVariances();
	}

	for (i = 0; i < NumberOfMovingChannels; i++) {
		/*
		std::cout << "means of class " << i+1 << ": " << GetMeanVector()->at(i) << std::endl;
		/*/
		int k;
		printf("means of class %s: ", label[i]);
		for (k = 0; k < NumberOfFixedChannels; k++) {
			printf("%f ", GetMeanVector()->at(i)[k]);
		}
		printf("\n");
		//*/
	}

#if 0
	{
		char means_file[1024];
		char variances_file[1024];
		sprintf(means_file, "jsr_means_%d.txt", this->GetNumberOfElapsedIterations());
		sprintf(variances_file, "jsr_variances_%d.txt", this->GetNumberOfElapsedIterations());
		SaveMeansAndVariances(means_file, variances_file, GetMeanVector(), GetVarianceVector());
	}
#endif

	{
		std::cout << "updating weight images ..." << std::endl;
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
		ComputeWeightImages(m_bUpdateMeansAndVariances);
		//ComputeWeightImages(true);
#else
		ComputeWeightImages(true);
#endif
	}

	//std::cout << "metric: " << GetMetric() << std::endl;
	
#if 0
	typedef itk::ImageFileWriter<FixedImageType> WriterType; 
	typename WriterType::Pointer writer_test = WriterType::New();
	typename itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();
	writer_test->SetImageIO(imageIO);

	std::cout << "writing fixed images..." << std::endl;
	for (i = 1; i <= NumberOfFixedChannels; i++) {
		char name[1024];
		sprintf(name,"jsr_fixed_image_%d_%d.nii.gz", this->GetNumberOfElapsedIterations(), i-1);
		writer_test->SetFileName(name);
		writer_test->SetInput(GetNthFixedImage(i));
		writer_test->Update();
	}

	std::cout << "writing weight images..." << std::endl;
	for (i = 1; i <= NumberOfMovingChannels; i++) {
		char name[1024];
		sprintf(name,"jsr_weight_image_%d_%d.nii.gz", this->GetNumberOfElapsedIterations(), i-1);
		writer_test->SetFileName(name);
		writer_test->SetInput(GetNthWeightImage(i));
		writer_test->Update();
	}

	std::cout << "writing warped images..." << std::endl;
	typedef itk::ChangeLabelImageFilter<MovingImageType, MovingImageType> ChangeLabelFilterType;
	typename ChangeLabelFilterType::Pointer change_filter = ChangeLabelFilterType::New();
	for (i = 1; i <= NumberOfMovingChannels; i++) {
		char name[1024];
		sprintf(name,"jsr_warped_image_%d_%d.nii.gz", this->GetNumberOfElapsedIterations(), i-1);
		writer_test->SetFileName(name);
		change_filter->SetInput(GetNthImageWarper(i)->GetOutput());
		change_filter->SetChange(NumericTraits<MovingPixelType>::max(), 0.0);
		writer_test->SetInput(change_filter->GetOutput());
		writer_test->Update();
	} 

#ifdef USE_PROB_PRIOR
	std::cout << "writing prob images..." << std::endl;
	for (i = 1; i <= NumberOfMovingChannels ; i++) {
		char name[1024];
		sprintf(name,"jsr_prob_image_%d_%d.nii.gz", this->GetNumberOfElapsedIterations(), i-1);
		writer_test->SetFileName(name);
		change_filter->SetInput(GetNthProbImage(i));
		writer_test->Update();
	} 
#endif
#endif

	// log the variables
	LogVariables();
	
	// initialize metric computation variables
	m_NumberOfPixelsProcessed = 0;
	m_SumOfEMLogs = 0;
	m_SumOfSquaredChange = 0;

	std::cout << "Initialize iteration done, computing updates ..." << std::endl;
}

// Computing the update in a non boundry neighborhood
// We skip over computing Hessians (because of its computations load)
// and update the deformation field using a Levenberg-Marquardt scheme.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
typename JointSegmentationRegistrationFunction<TFixedImage, TMovingImage, TDeformationField, VNumberOfFixedChannels, VNumberOfMovingChannels>
::PixelType
JointSegmentationRegistrationFunction<TFixedImage, TMovingImage, TDeformationField, VNumberOfFixedChannels, VNumberOfMovingChannels>
::ComputeUpdate(const NeighborhoodType &it, void * gd, const FloatOffsetType& itkNotUsed(offset))
{ 
	//cout << "compute update" << endl;
	GlobalDataStruct *globalData = (GlobalDataStruct*)gd;
	PixelType update;
	update.Fill(0.0);
	int i, j, k, dim;
	//return update;

	const IndexType index = it.GetIndex();
	IndexType FirstIndex = this->GetNthFixedImage(1)->GetLargestPossibleRegion().GetIndex();
	IndexType LastIndex = this->GetNthFixedImage(1)->GetLargestPossibleRegion().GetIndex() + this->GetNthFixedImage(1)->GetLargestPossibleRegion().GetSize();
	// Get moving image related information
	// check if the point was mapped outside of the moving image using
	// the "special value" NumericTraits<MovingPixelType>::max()
	double movingValues[NumberOfMovingChannels];
	double posteriorValues[NumberOfMovingChannels];
#ifdef USE_PROB_PRIOR
	double probValues[NumberOfMovingChannels];
#endif

	for (i = 1; i <= NumberOfMovingChannels; i++) {
		movingValues[i-1] = GetNthImageWarper(i)->GetOutput()->GetPixel(index);
		posteriorValues[i-1] = GetNthWeightImage(i)->GetPixel(index);
		if (movingValues[i-1] == NumericTraits <MovingPixelType>::max()) {
			update.Fill(0.0);
			return update;
		}
#if defined(USE_PROB_PRIOR) && !defined(PROB_LOAD_TU_ONLY)
		probValues[i-1] = GetNthProbImage(i)->GetPixel(index);
#endif
	}
#if defined(USE_PROB_PRIOR) && defined(PROB_LOAD_TU_ONLY)
	probValues[TU] = GetNthProbImage(TU+1)->GetPixel(index);
#endif
	
	// declare a variable to hold grad vectors
	CovariantVectorType warpedMovingGradients[NumberOfMovingChannels];
	MovingPixelType movingPixValues[NumberOfMovingChannels];

	// we don't use a CentralDifferenceImageFunction here to be able to
	// check for NumericTraits<MovingPixelType>::max()
	IndexType tmpIndex = index;

	// to the number of the dimensions in the images
	for (dim = 0; dim < ImageDimension; dim++)
	{
		// bounds checking
		if ((FirstIndex[dim] == LastIndex[dim]) || (index[dim] < FirstIndex[dim]) || (index[dim] >= LastIndex[dim]))
		{
			// case one: completely out of bounds
			for (i = 1; i < NumberOfMovingChannels + 1; i++) {
				warpedMovingGradients[i-1][dim] = 0.0;
			}
			continue;
		} else if (index[dim] == FirstIndex[dim]) {
			// case two: starting edge touch
			// compute derivative
			tmpIndex[dim] += 1;
			for (i = 1; i < NumberOfMovingChannels+1; i++) {
				movingPixValues[i-1] = GetNthImageWarper(i)->GetOutput()->GetPixel(tmpIndex);
				if (movingPixValues[i-1] == NumericTraits <MovingPixelType>::max()) {
					// weird crunched border case
					warpedMovingGradients[i-1][dim] = 0.0;
				} else {
					// forward difference
					warpedMovingGradients[i-1][dim] = static_cast<double>(movingPixValues[i-1]) - movingValues[i-1];
					warpedMovingGradients[i-1][dim] /= m_FixedImageSpacing[dim]; 
				}
			}
			tmpIndex[dim] -= 1;
			continue;
		} else if (index[dim] == (LastIndex[dim]-1)) {
			//case three: ending edge touch
			// compute derivative
			tmpIndex[dim] -= 1;
			// case three:
			for (i = 1; i < NumberOfMovingChannels+1; i++) {
				movingPixValues[i-1] = GetNthImageWarper(i)->GetOutput()->GetPixel(tmpIndex);
				if (movingPixValues[i-1] == NumericTraits<MovingPixelType>::max()) {
					// weird crunched border case
					warpedMovingGradients[i-1][dim] = 0.0;
				} else {
					// backward difference
					warpedMovingGradients[i-1][dim] = movingValues[i-1] - static_cast<double>(movingPixValues[i-1]);
					warpedMovingGradients[i-1][dim] /= m_FixedImageSpacing[dim]; 
				}
			}
			tmpIndex[dim] += 1;
			continue;
		} 

		tmpIndex[dim] += 1;

		for (i = 1; i < NumberOfMovingChannels+1; i++) {
			movingPixValues[i-1] = GetNthImageWarper(i)->GetOutput()->GetPixel(tmpIndex);
			if (movingPixValues[i-1] == NumericTraits<MovingPixelType>::max()) {
				// backward difference
				warpedMovingGradients[i-1][dim] = movingValues[i-1];
				tmpIndex[dim] -= 2;
				if (GetNthImageWarper(i)->GetOutput()->GetPixel(tmpIndex) == NumericTraits<MovingPixelType>::max()) {
					// weird crunched border case
					warpedMovingGradients[i-1][dim] = 0.0;
				} else {
					// backward difference
					warpedMovingGradients[i-1][dim] -= static_cast<double>(GetNthImageWarper(i)->GetOutput()->GetPixel(tmpIndex));
					warpedMovingGradients[i-1][dim] /= m_FixedImageSpacing[dim];
				}
				tmpIndex[dim] += 2;
			} else {
				warpedMovingGradients[i-1][dim] = static_cast<double>(movingPixValues[i-1]);
				tmpIndex[dim] -= 2;
				if (GetNthImageWarper(i)->GetOutput()->GetPixel(tmpIndex) == NumericTraits<MovingPixelType>::max()) {
					// forward difference
					warpedMovingGradients[i-1][dim] -= movingValues[i-1];
					warpedMovingGradients[i-1][dim] /= m_FixedImageSpacing[dim];
				} else {
					// normal case, central difference
					warpedMovingGradients[i-1][dim] -= static_cast<double>(GetNthImageWarper(i)->GetOutput()->GetPixel(tmpIndex));
					warpedMovingGradients[i-1][dim] *= 0.5 / m_FixedImageSpacing[dim];
				}
				tmpIndex[dim] += 2;
			}
		} // for number of channels

		tmpIndex[dim] -= 1;
	} // for dimensions

	// so far warpedMovingGradients have been calculated.

	// adding orientation informaiton
#if ITK_VERSION_MAJOR >= 4
	CovariantVectorType usedGradients[NumberOfMovingChannels];
	for (i = 1; i < NumberOfMovingChannels+1; i++) {
		this->GetNthFixedImage(1)->TransformLocalVectorToPhysicalVector(warpedMovingGradients[i-1], usedGradients[i-1]);
	}
#else
#ifdef ITK_USE_ORIENTED_IMAGE_DIRECTION
	CovariantVectorType usedGradients[NumberOfMovingChannels];
	for (i = 1; i < NumberOfMovingChannels+1; i++) {
		this->GetNthFixedImage(1)->TransformLocalVectorToPhysicalVector(warpedMovingGradients[i-1], usedGradients[i-1]);
	}
#else
	CovariantVectorType usedGradients[NumberOfMovingChannels];
	for (i = 1; i < NumberOfMovingChannels+1; i++) {
		usedGradients[i-1] = warpedMovingGradients[i-1];
	} 
#endif
#endif

#ifdef USE_FAST_LIKELIHOOD
	double gv[NumberOfMovingChannels][3];
	double Av[9];
	double bv[3];

	// assume 3 dimension
	if (ImageDimension != 3) {
		std::cout << "ImageDimension must be 3" << std::endl;
		return update;
	}
#endif

	// making vnl vectors for gradients
#ifdef USE_FAST_LIKELIHOOD
	for (i = 0; i < NumberOfMovingChannels; i++) {
		for (j = 0; j < 3; j++) {
			gv[i][j] = usedGradients[i][j];
		}
	}
#else
	std::vector<vnl_vector_fixed<double, ImageDimension>>  grads_vector; 
	for (i = 1; i < NumberOfMovingChannels+1; i++) {
		vnl_vector_fixed<double, ImageDimension> v(0.0);
		for (j = 0; j < ImageDimension; j++) {
			v(j) = usedGradients[i-1][j];
		}
		grads_vector.push_back(v);
	}
#endif

	// we solve Ax=b for x
	// making b
#ifdef USE_FAST_LIKELIHOOD
	for (j = 0; j < 3; j++) {
		bv[j] = 0;
	}
	for (i = 0; i < NumberOfMovingChannels; i++) {
		double pm;
#ifdef USE_JSR_CHECK_PRIOR_MAG
		if (movingValues[i] > 1e-3) {
			pm = posteriorValues[i] / movingValues[i];
			for (j = 0; j < 3; j++) {
				bv[j] += pm * gv[i][j];
			}
		}
#else
		//*
		pm = (posteriorValues[i] / (movingValues[i] + eps));
		/*/
		if (movingValues[i] < eps) {
			pm = 0;
		} else {
			pm = posteriorValues[i] / movingValues[i];
		}
		//*/
		for (j = 0; j < 3; j++) {
			bv[j] += pm * gv[i][j];
		}
#endif
	}
#else
	vnl_vector_fixed<double, ImageDimension> b(0.0);
	for (i = 0; i < NumberOfMovingChannels; i++) {
		b += (posteriorValues[i] / (movingValues[i] + eps)) * grads_vector[i];
	}
#endif
#ifdef USE_PROB_PRIOR
	{
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
		// debug: 9 -> 18??
#ifdef USE_JSR_PROB_WEIGHT_OLD
		double pm = -m_ProbWeight * 9 * (movingValues[TU] - probValues[TU]);
#else
		double pm = -2 * m_ProbWeight * 9 * (movingValues[TU] - probValues[TU]);
#endif
#else
		// debug: 4 -> 8??
		// multiplying 4 is considering tu_prior = prior[TU] + prior[NCR]
#ifdef USE_JSR_PROB_WEIGHT_OLD
		double pm = -m_ProbWeight * 4 * (movingValues[TU] - probValues[TU]);
#else
		double pm = -2 * m_ProbWeight * 4 * (movingValues[TU] - probValues[TU]);
#endif
		//double pm = -m_ProbWeight * (posteriorValues[TU] + posteriorValues[NCR]) * (movingValues[TU] - probValues[TU]);
#endif
		for (j = 0; j < 3; j++) {
			bv[j] += pm * gv[TU][j];
		}
	}
#endif

	// making A
#ifdef USE_FAST_LIKELIHOOD
	{
		double s;
		s = GetSigma2();
		// decreasing of sigma (depricated)
		//s *= (1.0 - float(this->GetNumberOfElapsedIterations()) * GetDeltaSigma2());
		for (j = 0; j < 3; j++) {
			for (i = 0; i < 3; i++) {
				if (j == i) {
					Av[j*3+i] = s;
				} else {
					Av[j*3+i] = 0;
				}
			}
		}
		for (k = 0; k < NumberOfMovingChannels; k++) {
			double pm;
#ifdef USE_JSR_CHECK_PRIOR_MAG
			if (movingValues[k] > 1e-3) {
				pm = 0.5 * posteriorValues[k] / (movingValues[k] * movingValues[k]);
				for (j = 0; j < 3; j++) {
					for (i = 0; i < 3; i++) {
						Av[j*3+i] += pm * gv[k][j] * gv[k][i];
					}
				}
			}
#else
			//*
			pm = (0.5 * posteriorValues[k] / (movingValues[k] + epss) / (movingValues[k] + eps));
			/*/
			if (movingValues[k] < eps) {
				pm = 0;
			} else {
				pm = 0.5 * posteriorValues[k] / (movingValues[k] * movingValues[k]);
			}
			//*/
			for (j = 0; j < 3; j++) {
				for (i = 0; i < 3; i++) {
					Av[j*3+i] += pm * gv[k][j] * gv[k][i];
				}
			}
#endif
		}
	}
#else
	vnl_matrix_fixed<double, ImageDimension, ImageDimension> A(0.0);
	// adding a null component to A
	A.set_identity();
	A *= GetSigma2();
	// decreasing of sigma (depricated)
	//A *= (1.0 - float(this->GetNumberOfElapsedIterations()) * GetDeltaSigma2());

	for (i = 0; i < NumberOfMovingChannels; i++) {
		A += (0.5*posteriorValues[i] / (movingValues[i] + epss) / (movingValues[i] + eps)) * outer_product(grads_vector[i], grads_vector[i]);
	}
#endif
#ifdef USE_PROB_PRIOR
	{
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
		double pm = m_ProbWeight * 9;
#else
		// multiplying 4 is considering tu_prior = prior[TU] + prior[NCR]
		double pm = m_ProbWeight * 4;
		//double pm = m_ProbWeight * (posteriorValues[TU] + posteriorValues[NCR]);
#endif
		for (j = 0; j < 3; j++) {
			for (i = 0; i < 3; i++) {
				Av[j*3+i] += pm * gv[TU][j] * gv[TU][i];
			}
		}
	}
#endif

	// compute update
	vnl_vector_fixed<double, ImageDimension> u;

#ifdef USE_FAST_LIKELIHOOD
	if (bv[0]*bv[0]+bv[1]*bv[1]+bv[2]*bv[2] > 1e-4) {
		double uv[3];
		SolveLinearSystem3(Av, bv, uv);
		u[0] = uv[0];
		u[1] = uv[1];
		u[2] = uv[2];
	} else {
		u[0] = 0;
		u[1] = 0;
		u[2] = 0;
	}
#else
	if (b.squared_magnitude() > 1e-4) {
		// this should be faster
		u = vnl_qr<double>(A).solve(b);
	} else {
		u.fill(0.0);
	}
#endif

	if (!u.is_finite()) {
		update.Fill(0.0);
		std::cout << "oops!, the update field is infinite!" << std::endl;
	} else {
		for (j = 0; j < ImageDimension; j++) {
			update[j] = u[j];
		}
	}

	if (globalData) {
		globalData->m_NumberOfPixelsProcessed += 1;
		globalData->m_SumOfSquaredChange += update.GetSquaredNorm();
	}

	// update done
	return update;
}

// Update the metric and release the per-thread-global data.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
JointSegmentationRegistrationFunction<TFixedImage, TMovingImage, TDeformationField, VNumberOfFixedChannels, VNumberOfMovingChannels>
::ReleaseGlobalDataPointer(void *gd) const
{
	GlobalDataStruct* globalData = (GlobalDataStruct*) gd;

	m_MetricCalculationLock.Lock();

	m_NumberOfPixelsProcessed += globalData->m_NumberOfPixelsProcessed;
	m_SumOfSquaredChange += globalData->m_SumOfSquaredChange;
	if (m_NumberOfPixelsProcessed) {
		m_RMSChange = vcl_sqrt(m_SumOfSquaredChange / static_cast<double>(m_NumberOfPixelsProcessed)); 
	}

	m_MetricCalculationLock.Unlock();

	delete globalData;
}


} // end namespace itk


#endif
