/**
 * @file  itkJointSegmentationRegistrationFilter.txx
 * @brief Filter class implementing the EM based joint segmentation-registration algorithm.
 *
 * Copyright (c) 2011-2014 University of Pennsylvania. All rights reserved.<br />
 * See http://www.cbica.upenn.edu/sbia/software/license.html or COPYING file.
 *
 * Contact: SBIA Group <sbia-software at uphs.upenn.edu>
 */

#ifndef __itkJointSegmentationRegistrationFilter_txx
#define __itkJointSegmentationRegistrationFilter_txx

#include <itkSmoothingRecursiveGaussianImageFilter.h>
#include "itkJointSegmentationRegistrationFilter.h"


namespace itk {


// Constructor/Destructor
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::JointSegmentationRegistrationFilter() : m_UseFirstOrderExp(false)
{
	std::cout << "filter constructor .." << std::endl;

	/** instantiating a function object and setting it up **/
	typename JointSegmentationRegistrationFunctionType::Pointer drfp;
	drfp = JointSegmentationRegistrationFunctionType::New();
	this->SetDifferenceFunction( static_cast<FiniteDifferenceFunctionType *>(drfp.GetPointer()));

	m_Multiplier = MultiplyByConstantType::New();
	m_Multiplier->InPlaceOn();
	m_MagnitudeCalculator = GradientToMagnitudeType::New();
	m_MinMaxCalculator = MinimumMaximumFieldType::New();
	m_Exponentiator = FieldExponentiatorType::New();

	m_Warper = VectorWarperType::New();
	FieldInterpolatorPointer VectorInterpolator = FieldInterpolatorType::New();
	m_Warper->SetInterpolator(VectorInterpolator);

	m_Adder = AdderType::New();
	m_Adder->InPlaceOn();

	std::cout <<"filter constructer done." << std::endl;
}

// Checks whether the DifferenceFunction is of type JointSegmentationRegistrationFunction.
// It throws and exception, if it is not.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
typename JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>::JointSegmentationRegistrationFunctionType *
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::DownCastDifferenceFunctionType()
{
	JointSegmentationRegistrationFunctionType *drfp = dynamic_cast<JointSegmentationRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());

	if (!drfp) {
		itkExceptionMacro(<< "Could not cast difference function to JointSegmentationRegistrationFunction");
	}

	return drfp;
}

// Checks whether the DifferenceFunction is of type JointSegmentationRegistrationFunction.
// It throws and exception, if it is not.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
const typename JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>::JointSegmentationRegistrationFunctionType *
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::DownCastDifferenceFunctionType() const
{
	const JointSegmentationRegistrationFunctionType *drfp = dynamic_cast<const JointSegmentationRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());

	if (!drfp) {
		itkExceptionMacro(<< "Could not cast difference function to JointSegmentationRegistrationFunction");
	}

	return drfp;
}
 
// Checks whether the DifferenceFunction is of type JointSegmentationRegistrationFunction.
// Set the function state values before each iteration.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::InitializeIteration()
{
	// update variables in the equation object
	JointSegmentationRegistrationFunctionType *f = this->DownCastDifferenceFunctionType();
#if ITK_VERSION_MAJOR >= 4
	f->SetDisplacementField(this->GetDisplacementField());
#else
	f->SetDeformationField(this->GetDeformationField());
#endif

	for (unsigned int i = 1; !this->GetElapsedIterations() && i <= NumberOfFixedChannels; i++) {
		std::cout << "setting fixed image " << i << " in fucntion .." << std::endl;
		f->SetNthFixedImage(GetNthFixedImage(i), i);
	}
	for (unsigned int i = 1; !this->GetElapsedIterations() && i <= NumberOfMovingChannels; i++) {
		std::cout << "setting moving image " << i << " in fucntion .." << std::endl;
		f->SetNthMovingImage(GetNthMovingImage(i), i);
	}
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	for (unsigned int i = 1; !this->GetElapsedIterations() && i <= NumberOfMovingChannels; i++) {
		std::cout << "setting weight image " << i << " in fucntion .." << std::endl;
		f->SetNthWeightImage(GetNthWeightImage(i), i);
	}
#endif
#ifdef USE_PROB_PRIOR
	for (unsigned int i = 1; !this->GetElapsedIterations() && i <= NumberOfMovingChannels; i++) {
		std::cout << "setting prob image " << i << " in fucntion .." << std::endl;
		f->SetNthProbImage(GetNthProbImage(i), i);
	}
#endif

	f->SetNumberOfMaximumIterations(this->GetNumberOfIterations());

	f->SetNumberOfElapsedIterations(this->GetElapsedIterations());
	f->InitializeIteration();

#if defined(USE_MEAN_SHIFT_UPDATE)
	if (this->GetElapsedIterations()) {
		f->m_bMeanShiftUpdate = true;
	} else {
		f->m_bMeanShiftUpdate = false;
	}
#elif defined(USE_MEAN_SHIFT_UPDATE_ONCE)
	f->m_bMeanShiftUpdate = false;
#endif

	std::cout <<"iteration: " << this->GetElapsedIterations() << std::endl;
}

// Returns a (const) pointer to the Nth fixed image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
const typename JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::FixedImageType*
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthFixedImage(unsigned int n) const
{
	return dynamic_cast< const FixedImageType * >(this->ProcessObject::GetInput(n));
}
// Returns a pointer to the Nth fixed image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
typename JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::FixedImageType*
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthFixedImage(unsigned int n) 
{
	return dynamic_cast< FixedImageType * >(this->ProcessObject::GetInput(n));
}
// Sets a (const) pointer to the Nth fixed image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::SetNthFixedImage(const FixedImageType * ptr, unsigned int n )
{
	std::cout << "setting fixed image " << n << " in the filter." << std::endl;
	this->ProcessObject::SetNthInput(n, const_cast< FixedImageType * >(ptr));
}

// Returns a (const) pointer of the Nth moving image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
const typename JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::MovingImageType *
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthMovingImage(unsigned int n) const
{
	return dynamic_cast< const MovingImageType * >(this->ProcessObject::GetInput(NumberOfFixedChannels + n)); 
}
// Returns a pointer of the Nth moving image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
typename JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::MovingImageType *
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthMovingImage(unsigned int n)
{
	return dynamic_cast< MovingImageType * >(this->ProcessObject::GetInput(NumberOfFixedChannels + n));
}
// Sets a (const) pointer of the Nth moving image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::SetNthMovingImage(const MovingImageType * ptr, unsigned int n )
{
	std::cout << "setting moving prior image " << n << " in the filter." << std::endl;
	this->ProcessObject::SetNthInput(NumberOfFixedChannels + n, const_cast< MovingImageType * >(ptr));
}

#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
// Returns a (const) pointer of the Nth prob image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
const typename JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::WeightImageType *
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthWeightImage(unsigned int n) const
{
	return dynamic_cast< const WeightImageType * >(this->ProcessObject::GetInput( NumberOfFixedChannels + NumberOfMovingChannels + n)); 
}
// Returns a pointer of the Nth prob image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
typename JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::WeightImageType *
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthWeightImage(unsigned int n)
{
	return dynamic_cast< WeightImageType * >(this->ProcessObject::GetInput( NumberOfFixedChannels + NumberOfMovingChannels + n)); 
}
// Sets a (const) pointer of the Nth prob image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::SetNthWeightImage(const WeightImageType * ptr, unsigned int n )
{
	std::cout << "setting weight image " << n << " in the filter." << std::endl;
	this->ProcessObject::SetNthInput(NumberOfFixedChannels + NumberOfMovingChannels + n, const_cast< WeightImageType * >(ptr));
}
#else
// Returns a pointer to the Nth posterior image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
typename JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::MovingImageType*
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthWeightImage(unsigned int n) 
{
	JointSegmentationRegistrationFunctionType *drfp = dynamic_cast<JointSegmentationRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());
	if (n <= 0 || NumberOfMovingChannels < n) {
		std ::cout << "invalid weight image is requested " << std::endl;
	} else {
		return (drfp->GetNthWeightImage(n));
	}
	return NULL;
}
#endif

#ifdef USE_PROB_PRIOR
// Returns a (const) pointer of the Nth prob image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
const typename JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::ProbImageType *
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthProbImage(unsigned int n) const
{
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	return dynamic_cast< const ProbImageType * >(this->ProcessObject::GetInput( NumberOfFixedChannels + NumberOfMovingChannels + NumberOfMovingChannels + n)); 
#else
	return dynamic_cast< const ProbImageType * >(this->ProcessObject::GetInput( NumberOfFixedChannels + NumberOfMovingChannels + n)); 
#endif
}
// Returns a pointer of the Nth prob image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
typename JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::ProbImageType *
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthProbImage(unsigned int n)
{
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	return dynamic_cast< ProbImageType * >(this->ProcessObject::GetInput( NumberOfFixedChannels + NumberOfMovingChannels + NumberOfMovingChannels + n)); 
#else
	return dynamic_cast< ProbImageType * >(this->ProcessObject::GetInput( NumberOfFixedChannels + NumberOfMovingChannels + n)); 
#endif
}
// Sets a (const) pointer of the Nth prob image.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::SetNthProbImage(const ProbImageType * ptr, unsigned int n )
{
	std::cout << "setting prob image " << n << " in the filter." << std::endl;
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	this->ProcessObject::SetNthInput(NumberOfFixedChannels + NumberOfMovingChannels + NumberOfMovingChannels + n, const_cast< ProbImageType * >(ptr));
#else
	this->ProcessObject::SetNthInput(NumberOfFixedChannels + NumberOfMovingChannels + n, const_cast< ProbImageType * >(ptr));
#endif
}
#endif


template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::SetUpdateMeansAndVariances(bool bUpdateMeansAndVariances)
{
	JointSegmentationRegistrationFunctionType *drfp = dynamic_cast<JointSegmentationRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());
	drfp->m_bUpdateMeansAndVariances = bUpdateMeansAndVariances;
}
//
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
bool
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetUpdateMeansAndVariances()
{
	JointSegmentationRegistrationFunctionType *drfp = dynamic_cast<JointSegmentationRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());
	return drfp->m_bUpdateMeansAndVariances;
}


// Returns the number of valid inputs
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
std::vector<SmartPointer<DataObject> >::size_type
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNumberOfValidRequiredInputs() const
{
	typename std::vector<SmartPointer<DataObject> >::size_type num = 0;

	for (unsigned int i = 1; i <= NumberOfFixedChannels; i++) {
		if (GetNthFixedImage(i)) {
			num++;
		}
	}
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		if (GetNthMovingImage(i)) {
			num++;
		}
	}
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		if (GetNthWeightImage(i)) {
			num++;
		}
	}
#endif
#ifdef USE_PROB_PRIOR
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		if (GetNthProbImage(i)) {
			num++;
		}
	}
#endif

	return num;
}

// Get the metric value from the difference function
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
double
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetMetric() const
{
	const JointSegmentationRegistrationFunctionType *drfp = this->DownCastDifferenceFunctionType();
	return drfp->GetMetric();
}

// Get the RMS change from the difference function
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
const double & 
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetRMSChange() const
{
	const JointSegmentationRegistrationFunctionType *drfp = this->DownCastDifferenceFunctionType();
	return drfp->GetRMSChange();
}

// Allocate an update buffer for the filter and also posterior images
// in the functions.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::AllocateUpdateBuffer()
{
	// The update buffer looks just like the output.
	DeformationFieldPointer output = this->GetOutput();
	DeformationFieldPointer upbuf = this->GetUpdateBuffer();
	std::cout << "AllocateUpdateBuffer ia called." << std::endl;
	upbuf->SetLargestPossibleRegion(output->GetLargestPossibleRegion());
	upbuf->SetRequestedRegion(output->GetRequestedRegion());
	upbuf->SetBufferedRegion(output->GetBufferedRegion());
	upbuf->SetOrigin(output->GetOrigin());
	upbuf->SetSpacing(output->GetSpacing());
	upbuf->SetDirection(output->GetDirection());
	upbuf->Allocate();
	
#ifndef USE_JSR_WEGHT_IMAGE_PYRAMID
	std::cout << "Allocating weight images in the function object ..." << std::endl;

	JointSegmentationRegistrationFunctionType *drfp = this->DownCastDifferenceFunctionType();

	for (int i = 1; i <= NumberOfMovingChannels; i++) {
		drfp->GetNthWeightImage(i)->SetRegions(output->GetLargestPossibleRegion());
		drfp->GetNthWeightImage(i)->SetOrigin(output->GetOrigin());
		drfp->GetNthWeightImage(i)->SetSpacing(output->GetSpacing());
		drfp->GetNthWeightImage(i)->SetDirection(output->GetDirection());
		drfp->GetNthWeightImage(i)->Allocate(); 
	}
#endif
}

// Smoothes and applies the deformation field computed by the function class
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
#if ITK_VERSION_MAJOR >= 4
::ApplyUpdate(const TimeStepType& dt)
#else
::ApplyUpdate(TimeStepType dt)
#endif
{
	std::cout << "applying update in filter .." << std::endl;
	std::cout << "dt = " << dt << std::endl;

	// If we smooth the update buffer before applying it, then the are
	// approximating a viscuous problem as opposed to an elastic problem
	if (this->GetSmoothUpdateField()) {
		/**smoothing update field**/
		this->SmoothUpdateField();
	}

#ifdef USE_JSR_NORMALIZE_UPDATE_FIELD
	// normalize update field
	{
		double max_w, max_w_1;

		m_MagnitudeCalculator->SetInput(this->GetUpdateBuffer());
		m_MagnitudeCalculator->Update();

		m_MinMaxCalculator->SetImage(m_MagnitudeCalculator->GetOutput());
		m_MinMaxCalculator->ComputeMaximum();
		max_w = m_MinMaxCalculator->GetMaximum();
		max_w_1 = 1.0 / max_w;
		std::cout << "max_w = " << max_w << std::endl;

		m_Multiplier->SetConstant(max_w_1);
		m_Multiplier->SetInput(this->GetUpdateBuffer());
		m_Multiplier->GraftOutput(this->GetUpdateBuffer());
		// in place update
		m_Multiplier->Update();
		// graft output back to this->GetUpdateBuffer()
		this->GetUpdateBuffer()->Graft(m_Multiplier->GetOutput());

		/*
		m_MagnitudeCalculator->SetInput(this->GetUpdateBuffer());
		m_MagnitudeCalculator->Update();

		m_MinMaxCalculator->SetImage(m_MagnitudeCalculator->GetOutput());
		m_MinMaxCalculator->ComputeMaximum();
		max_w = m_MinMaxCalculator->GetMaximum();
		std::cout << "max_w = " << max_w << std::endl;
		//*/
	}
#endif

	// Use time step if necessary. In many cases
	// the time step is one so this will be skipped
	if (fabs(dt - 1.0) > 1.0e-4) {
		itkDebugMacro("Using timestep: " << dt);
		m_Multiplier->SetConstant(dt);
		m_Multiplier->SetInput(this->GetUpdateBuffer());
		m_Multiplier->GraftOutput(this->GetUpdateBuffer());
		// in place update
		m_Multiplier->Update();
		// graft output back to this->GetUpdateBuffer()
		this->GetUpdateBuffer()->Graft(m_Multiplier->GetOutput());
	}

	if (this->m_UseFirstOrderExp) {
		// use s <- s o (Id +u)

		// skip exponential and compose the vector fields
		m_Warper->SetOutputOrigin(this->GetUpdateBuffer()->GetOrigin());
		m_Warper->SetOutputSpacing(this->GetUpdateBuffer()->GetSpacing());
		m_Warper->SetOutputDirection(this->GetUpdateBuffer()->GetDirection());
		m_Warper->SetInput(this->GetOutput());
#if ITK_VERSION_MAJOR >= 4
		m_Warper->SetDisplacementField(this->GetUpdateBuffer());
#else
		m_Warper->SetDeformationField(this->GetUpdateBuffer());
#endif

		m_Adder->SetInput1(m_Warper->GetOutput());
		m_Adder->SetInput2(this->GetUpdateBuffer());

		m_Adder->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
	} else {
		// use s <- s o exp(u)

		// compute the exponential
		m_Exponentiator->SetInput(this->GetUpdateBuffer());

		//const double imposedMaxUpStep = this->GetMaximumUpdateStepLength();
		const double imposedMaxUpStep = 2.0;
		if (imposedMaxUpStep > 0.0) {
			// max(norm(Phi))/2^N <= 0.25*pixelspacing
			const double numiterfloat = 2.0 + vcl_log(imposedMaxUpStep)/vnl_math::ln2;
			unsigned int numiter = 0;
			if (numiterfloat > 0.0) {
				numiter = static_cast<unsigned int>(vcl_ceil(numiterfloat));
			}

			m_Exponentiator->AutomaticNumberOfIterationsOff();
			m_Exponentiator->SetMaximumNumberOfIterations(numiter);
		} else {
			m_Exponentiator->AutomaticNumberOfIterationsOn();
			// just set a high value so that automatic number of step
			// is not thresholded
			m_Exponentiator->SetMaximumNumberOfIterations(2000u);
		}

		m_Exponentiator->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());

		m_Exponentiator->Update();

		// compose the vector fields
		m_Warper->SetOutputOrigin(this->GetUpdateBuffer()->GetOrigin());
		m_Warper->SetOutputSpacing(this->GetUpdateBuffer()->GetSpacing());
		m_Warper->SetOutputDirection(this->GetUpdateBuffer()->GetDirection());
		m_Warper->SetInput(this->GetOutput());
#if ITK_VERSION_MAJOR >= 4
		m_Warper->SetDisplacementField(m_Exponentiator->GetOutput());
#else
		m_Warper->SetDeformationField(m_Exponentiator->GetOutput());
#endif

		m_Warper->Update();

		m_Adder->SetInput1(m_Warper->GetOutput());
		m_Adder->SetInput2(m_Exponentiator->GetOutput());

		m_Adder->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
	}

	// Triggers update
	m_Adder->Update();

	// Region passing stuff
	this->GraftOutput(m_Adder->GetOutput());
	this->GetOutput()->Modified();

	JointSegmentationRegistrationFunctionType *drfp = this->DownCastDifferenceFunctionType();

	this->SetRMSChange(drfp->GetRMSChange());

	std::cout << "smoothing deformation field .." << std::endl;
	/**
	 * Smooth the deformation field
	 */
#if ITK_VERSION_MAJOR >= 4
	if (this->GetSmoothDisplacementField()) {
		this->SmoothDisplacementField();
	}
#else
	if (this->GetSmoothDeformationField()) {
		this->SmoothDeformationField();
	}
#endif
	std::cout << "smoothing deformation done.." << std::endl;

#if defined(USE_MEAN_SHIFT_UPDATE_ONCE)
	if (this->GetElapsedIterations() == this->GetNumberOfIterations()-1) {
		std::cout << "update means and variances using mean shift" << std::endl;
		drfp->m_bMeanShiftUpdate = true;
		drfp->UpdateMeansAndVariances();
	} else {
		drfp->m_bMeanShiftUpdate = false;
	}
#endif
}

// Standard PrintSelf function.
template <class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
JointSegmentationRegistrationFilter<TFixedImage,TMovingImage,TDeformationField,VNumberOfFixedChannels,VNumberOfMovingChannels>
::PrintSelf(std::ostream& os, Indent indent) const
{ 
	Superclass::PrintSelf(os, indent);
	os << indent << "Use First Order exponential: " << this->m_UseFirstOrderExp << std::endl;
}


} // end namespace itk


#endif
