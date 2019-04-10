/**
 * @file  itkMultiResolutionJointSegmentationRegistration.txx
 * @brief Filter class implementing multi resolution joint segmentation-registration algorithm.
 *
 * Copyright (c) 2011-2014 University of Pennsylvania. All rights reserved.<br />
 * See http://www.cbica.upenn.edu/sbia/software/license.html or COYPING file.
 *
 * Contact: SBIA Group <sbia-software at uphs.upenn.edu>
 */

#ifndef _itkMultiResolutionJointSegmentationRegistration_txx
#define _itkMultiResolutionJointSegmentationRegistration_txx

#include <itkRecursiveGaussianImageFilter.h>
#include <itkRecursiveMultiResolutionPyramidImageFilter.h>
#include <itkImageRegionIterator.h>
#include <vnl/vnl_math.h>

#include "itkMultiResolutionJointSegmentationRegistration.h"


namespace itk {


// Default constructor
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::MultiResolutionJointSegmentationRegistration()
{
	std::cout << "Multi resolution filter constructor .." << std::endl;

	/** setting the total number of required inputs**/
	{
		int num_imputs = NumberOfFixedChannels + NumberOfMovingChannels;
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
		num_imputs += NumberOfMovingChannels;
#endif
#ifdef USE_PROB_PRIOR
		num_imputs += NumberOfMovingChannels;
#endif
		this->SetNumberOfRequiredInputs(num_imputs);
	}

#if ITK_VERSION_MAJOR >= 4
	this->RemoveRequiredInputName( "Primary" );
#endif

	/** instantiating a segmentor_registrator filter object **/
	typename DefaultSegmentationRegistrationType::Pointer segmentor_registrator = DefaultSegmentationRegistrationType::New();
	m_SegmentationRegistrationFilter = static_cast<DefaultSegmentationRegistrationType*>(segmentor_registrator.GetPointer());

	/**determining the number of levels **/
	m_NumberOfLevels = 3;
	m_NumberOfIterations.resize(m_NumberOfLevels);

	m_SegmentationRegistrationFilter = DefaultSegmentationRegistrationType::New();

	/**allocating the number of channels in moving and fixed pyramids**/
	m_MovingImagePyramidVector.resize(NumberOfMovingChannels);
	m_FixedImagePyramidVector.resize(NumberOfFixedChannels);
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	m_WeightImagePyramidVector.resize(NumberOfMovingChannels);
#endif
#ifdef USE_PROB_PRIOR
	m_ProbImagePyramidVector.resize(NumberOfMovingChannels);
#endif

	for (unsigned int i = 1; i <= NumberOfFixedChannels; i++) {
		m_FixedImagePyramidVector[i-1] = FixedImagePyramidType::New();
		m_FixedImagePyramidVector[i-1]->SetNumberOfLevels(m_NumberOfLevels);
	}
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		m_MovingImagePyramidVector[i-1] = MovingImagePyramidType::New();
		m_MovingImagePyramidVector[i-1]->SetNumberOfLevels(m_NumberOfLevels);
	}
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		m_WeightImagePyramidVector.at(i-1) = WeightImagePyramidType::New();
		m_WeightImagePyramidVector.at(i-1)->SetNumberOfLevels(m_NumberOfLevels);
	}
#endif
#ifdef USE_PROB_PRIOR
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		m_ProbImagePyramidVector[i-1] = ProbImagePyramidType::New();
		m_ProbImagePyramidVector[i-1]->SetNumberOfLevels(m_NumberOfLevels);
	}
#endif

	m_FieldExpander = FieldExpanderType::New();
	m_InitialDeformationField = NULL;
	
	/** initializing the number of levels **/
	unsigned int ilevel;
	for (ilevel = 0; ilevel < m_NumberOfLevels; ilevel++) {
		m_NumberOfIterations[ilevel] = 10;
	}
	m_CurrentLevel = 0;

	m_StopRegistrationFlag = false;
	std::cout << "Multi resolution filter constructor done .." << std::endl;
}

// Set the nth fixed image.
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::SetNthFixedImage(const FixedImageType * ptr, unsigned int n)
{
	this->ProcessObject::SetNthInput(n, const_cast< FixedImageType * >(ptr));
}
// Get the nth fixed image.
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
const typename MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::FixedImageType *
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthFixedImage(unsigned int n) const
{
	return dynamic_cast< const FixedImageType * >(this->ProcessObject::GetInput(n));
}

// Set the moving image image.
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::SetNthMovingImage(const MovingImageType * ptr, unsigned int n)
{
	this->ProcessObject::SetNthInput(NumberOfFixedChannels + n, const_cast< MovingImageType * >(ptr));
}
// Get the moving image image.
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
const typename MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType, VNumberOfFixedChannels,VNumberOfMovingChannels>
::MovingImageType *
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthMovingImage(unsigned int n) const
{
	return dynamic_cast< const MovingImageType * >(this->ProcessObject::GetInput(NumberOfFixedChannels + n));
}

#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
// Set the prob image image.
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::SetNthWeightImage(const ProbImageType * ptr, unsigned int n)
{
	this->ProcessObject::SetNthInput(NumberOfFixedChannels + NumberOfMovingChannels + n, const_cast< WeightImageType * >(ptr));
}
// Get the prob image image.
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
const typename MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType, VNumberOfFixedChannels,VNumberOfMovingChannels>
::WeightImageType *
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthWeightImage(unsigned int n) const
{
	return dynamic_cast< const WeightImageType * >(this->ProcessObject::GetInput(NumberOfFixedChannels + NumberOfMovingChannels + n));
}
#endif

#ifdef USE_PROB_PRIOR
// Set the prob image image.
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::SetNthProbImage(const ProbImageType * ptr, unsigned int n)
{
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	this->ProcessObject::SetNthInput(NumberOfFixedChannels + NumberOfMovingChannels + NumberOfMovingChannels + n, const_cast< ProbImageType * >(ptr));
#else
	this->ProcessObject::SetNthInput(NumberOfFixedChannels + NumberOfMovingChannels + n, const_cast< ProbImageType * >(ptr));
#endif
}
// Get the prob image image.
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
const typename MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType, VNumberOfFixedChannels,VNumberOfMovingChannels>
::ProbImageType *
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GetNthProbImage(unsigned int n) const
{
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	return dynamic_cast< const ProbImageType * >(this->ProcessObject::GetInput(NumberOfFixedChannels + NumberOfMovingChannels + NumberOfMovingChannels + n));
#else
	return dynamic_cast< const ProbImageType * >(this->ProcessObject::GetInput(NumberOfFixedChannels + NumberOfMovingChannels + n));
#endif
}
#endif

// Retrives the number of required inputs.
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
std::vector<SmartPointer<DataObject> >::size_type
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
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

// Set the number of multi-resolution levels
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::SetNumberOfLevels(unsigned int num)
{
	if (m_NumberOfLevels != num) {
		this->Modified();
		m_NumberOfLevels = num;
		m_NumberOfIterations.resize(m_NumberOfLevels);
	}

	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		m_MovingImagePyramidVector[i-1]->SetNumberOfLevels(m_NumberOfLevels);
	}

	for(unsigned int i = 1; i <= NumberOfFixedChannels; i++) {
		m_FixedImagePyramidVector[i-1]->SetNumberOfLevels(m_NumberOfLevels);
	}

#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	for(unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		m_WeightImagePyramidVector[i-1]->SetNumberOfLevels(m_NumberOfLevels);
	}
#endif

#ifdef USE_PROB_PRIOR
	for(unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		m_ProbImagePyramidVector[i-1]->SetNumberOfLevels(m_NumberOfLevels);
	}
#endif
}

// Standard PrintSelf method.
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::PrintSelf(std::ostream& os, Indent indent) const
{
	Superclass::PrintSelf(os, indent);
	os << indent << "NumberOfLevels: " << m_NumberOfLevels << std::endl;
	os << indent << "CurrentLevel: " << m_CurrentLevel << std::endl;

	os << indent << "NumberOfIterations: [";
	unsigned int ilevel;
	for (ilevel = 0; ilevel < m_NumberOfLevels-1; ilevel++) {
		os << m_NumberOfIterations[ilevel] << ", ";
	}
	os << m_NumberOfIterations[ilevel] << "]" << std::endl;
	
	os << indent << "SegmentationRegistrationFilter: ";
	os << m_SegmentationRegistrationFilter.GetPointer() << std::endl;
	
	os << indent << "FixedImagePyramid: ";
	for (unsigned int i = 1; i <= NumberOfFixedChannels; i++) {
		os << m_FixedImagePyramidVector[i-1] << std::endl;
	}

	os << indent << "MovingImagePyramid: ";
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		os << m_MovingImagePyramidVector[i-1] << std::endl;
	}

#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	os << indent << "WeightImagePyramid: ";
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		os << m_WeightImagePyramidVector.at(i-1) << std::endl;
	}
#endif

#ifdef USE_PROB_PRIOR
	os << indent << "ProbImagePyramid: ";
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		os << m_ProbImagePyramidVector[i-1] << std::endl;
	}
#endif

	os << indent << "FieldExpander: ";
	os << m_FieldExpander.GetPointer() << std::endl;

	os << indent << "StopRegistrationFlag: ";
	os << m_StopRegistrationFlag << std::endl;
}

// Perform a the deformable registration using a multiresolution scheme
// using an internal mini-pipeline
//
//  ref_pyramid ->  segmentor_registrator  ->  field_expander --|| tempField
// test_pyramid ->           |                              |
//                           |                              |
//                           --------------------------------    
//
// A tempField image is used to break the cycle between the
// segmentor_registrator and field_expander.
template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GenerateData()
{
	// Check for NULL images and pointers
	if (!m_SegmentationRegistrationFilter) {
		itkExceptionMacro(<< "Registration filter not set");
	}

	if (this->m_InitialDeformationField && this->GetInput(0)) {
		itkExceptionMacro(<< "Only one initial deformation can be given. "
			<< "SetInitialDeformationField should not be used in "
			<< "cunjunction with SetArbitraryInitialDeformationField "
			<< "or SetInput.");
	}

	for (unsigned int i = 1; i <= NumberOfFixedChannels; i++) {
		FixedImageConstPointer fixedImage = GetNthFixedImage(i);

		if (!fixedImage) {
			itkExceptionMacro(<< i << "th fixed image not set");
		}
		if (!GetNthFixedImagePyramid(i)) {
			itkExceptionMacro(<< i << " th fixed pyramid not set");
		}

		std::cout << "Creating fixed pyramid number: " << i << std::endl;
		GetNthFixedImagePyramid(i)->SetInput(fixedImage);
		GetNthFixedImagePyramid(i)->UpdateLargestPossibleRegion();
	} // fixed for channels

	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		MovingImageConstPointer movingImage = GetNthMovingImage(i);

		if (!movingImage) {
			itkExceptionMacro(<< i << "th moving images not set");
		}
		if (!GetNthMovingImagePyramid(i)) {
			itkExceptionMacro(<< i << " th moving pyramid not set");
		}

		std::cout << "Creating moving pyramid number: " << i << std::endl;
		// Create the image pyramids.
		GetNthMovingImagePyramid(i)->SetInput(movingImage);
		GetNthMovingImagePyramid(i)->UpdateLargestPossibleRegion();
	} // for moving channels

#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		WeightImageConstPointer weightImage = GetNthWeightImage(i);

		if (!weightImage) {
			itkExceptionMacro(<< i << "th weight images not set");
		}
		if (!GetNthWeightImagePyramid(i)) {
			itkExceptionMacro(<< i << " th weight pyramid not set");
		}

		std::cout << "Creating weight pyramid number: " << i << std::endl;
		// Create the image pyramids.
		GetNthWeightImagePyramid(i)->SetInput(weightImage);
		GetNthWeightImagePyramid(i)->UpdateLargestPossibleRegion();
	} // for prob channels
#endif

#ifdef USE_PROB_PRIOR
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		ProbImageConstPointer probImage = GetNthProbImage(i);

		if (!probImage) {
			itkExceptionMacro(<< i << "th prob images not set");
		}
		if (!GetNthProbImagePyramid(i)) {
			itkExceptionMacro(<< i << " th prob pyramid not set");
		}

		std::cout << "Creating prob pyramid number: " << i << std::endl;
		// Create the image pyramids.
		GetNthProbImagePyramid(i)->SetInput(probImage);
		GetNthProbImagePyramid(i)->UpdateLargestPossibleRegion();
	} // for prob channels
#endif

	std::cout << "pyramids created." << std::endl;

	// Initializations
	m_CurrentLevel = 0;
	m_StopRegistrationFlag = false;

	unsigned int fixedLevel  = vnl_math_min((int)m_CurrentLevel, (int)GetNthFixedImagePyramid(1)->GetNumberOfLevels());
	unsigned int movingLevel = vnl_math_min((int)m_CurrentLevel, (int)GetNthMovingImagePyramid(1)->GetNumberOfLevels());
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	unsigned int weightLevel = vnl_math_min((int)m_CurrentLevel, (int)GetNthWeightImagePyramid(1)->GetNumberOfLevels());
#endif
#ifdef USE_PROB_PRIOR
	unsigned int probLevel   = vnl_math_min((int)m_CurrentLevel, (int)GetNthProbImagePyramid(1)->GetNumberOfLevels());
#endif

	DeformationFieldPointer tempField = NULL;
	DeformationFieldPointer inputPtr = const_cast< DeformationFieldType * >(this->GetInput(0));

	if (this->m_InitialDeformationField) {
		tempField = this->m_InitialDeformationField;
	} else if (inputPtr) {
		// Arbitrary initial deformation field is set.
		// smooth it and resample

		// First smooth it
		tempField = inputPtr;

		typedef RecursiveGaussianImageFilter< DeformationFieldType, DeformationFieldType> GaussianFilterType;
		typename GaussianFilterType::Pointer smoother = GaussianFilterType::New();

		for (unsigned int dim = 0; dim < DeformationFieldType::ImageDimension; ++dim) {
			// sigma accounts for the subsampling of the pyramid
			double sigma = 0.5 * static_cast<float>(GetNthFixedImagePyramid(1)->GetSchedule()[fixedLevel][dim]);

			// but also for a possible discrepancy in the spacing
			sigma *= GetNthFixedImage(1)->GetSpacing()[dim] / inputPtr->GetSpacing()[dim];

			smoother->SetInput(tempField);
			smoother->SetSigma(sigma);
			smoother->SetDirection(dim);

			smoother->Update();

			tempField = smoother->GetOutput();
			tempField->DisconnectPipeline();
		}

		// Now resample
		m_FieldExpander->SetInput(tempField);

		typename FloatImageType::Pointer fi = GetNthFixedImagePyramid(1)->GetOutput(fixedLevel);
		m_FieldExpander->SetSize(fi->GetLargestPossibleRegion().GetSize());
		m_FieldExpander->SetOutputStartIndex(fi->GetLargestPossibleRegion().GetIndex());
		m_FieldExpander->SetOutputOrigin(fi->GetOrigin());
		m_FieldExpander->SetOutputSpacing(fi->GetSpacing());
		m_FieldExpander->SetOutputDirection(fi->GetDirection());

		m_FieldExpander->UpdateLargestPossibleRegion();
		m_FieldExpander->SetInput(NULL);
		tempField = m_FieldExpander->GetOutput();
		tempField->DisconnectPipeline();
	}

	bool lastShrinkFactorsAllOnes = false;

	while (!this->Halt()) {
		if (tempField.IsNull()) {
#if ITK_VERSION_MAJOR >= 4
			m_SegmentationRegistrationFilter->SetInitialDisplacementField(NULL);
#else
			m_SegmentationRegistrationFilter->SetInitialDeformationField(NULL);
#endif
		} else {
			// Resample the field to be the same size as the fixed image
			// at the current level
			m_FieldExpander->SetInput(tempField);

			typename FloatImageType::Pointer fi = GetNthFixedImagePyramid(1)->GetOutput(fixedLevel);
			m_FieldExpander->SetSize(fi->GetLargestPossibleRegion().GetSize());
			m_FieldExpander->SetOutputStartIndex(fi->GetLargestPossibleRegion().GetIndex());
			m_FieldExpander->SetOutputOrigin(fi->GetOrigin());
			m_FieldExpander->SetOutputSpacing(fi->GetSpacing());
			m_FieldExpander->SetOutputDirection(fi->GetDirection());

			m_FieldExpander->UpdateLargestPossibleRegion();
			m_FieldExpander->SetInput(NULL);
			tempField = m_FieldExpander->GetOutput();
			tempField->DisconnectPipeline();

#if ITK_VERSION_MAJOR >= 4
			m_SegmentationRegistrationFilter->SetInitialDisplacementField(tempField);
#else
			m_SegmentationRegistrationFilter->SetInitialDeformationField(tempField);
#endif
		}

		for (unsigned int i = 1; i <= NumberOfFixedChannels; i++) {
			// setup registration filter and pyramids 
			m_SegmentationRegistrationFilter->SetNthFixedImage(GetNthFixedImagePyramid(i)->GetOutput(fixedLevel), i);
		}
		for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
			// setup registration filter and pyramids 
			m_SegmentationRegistrationFilter->SetNthMovingImage(GetNthMovingImagePyramid(i)->GetOutput(movingLevel), i);
		}
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
		for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
			// setup registration filter and pyramids 
			m_SegmentationRegistrationFilter->SetNthWeightImage(GetNthWeightImagePyramid(i)->GetOutput(weightLevel), i);
		}
#endif
#ifdef USE_PROB_PRIOR
		for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
			// setup registration filter and pyramids 
			m_SegmentationRegistrationFilter->SetNthProbImage(GetNthProbImagePyramid(i)->GetOutput(probLevel), i);
		}
#endif
		
		m_SegmentationRegistrationFilter->SetNumberOfIterations(m_NumberOfIterations[m_CurrentLevel]);

		// cache shrink factors for computing the next expand factors.
		lastShrinkFactorsAllOnes = true;
		for (unsigned int idim = 0; idim < ImageDimension; idim++) {
			if (GetNthFixedImagePyramid(1)->GetSchedule()[fixedLevel][idim] > 1) {
				lastShrinkFactorsAllOnes = false;
				break;
			}
		}

		if (m_CurrentLevel < m_NumberOfLevels-1) {
			m_SegmentationRegistrationFilter->SetUpdateMeansAndVariances(false);
		} else {
			m_SegmentationRegistrationFilter->SetUpdateMeansAndVariances(true);
		}

		// compute new deformation field
		std::cout << "start " << std::endl;
		m_SegmentationRegistrationFilter->UpdateLargestPossibleRegion();
		std::cout << "end " << std::endl;
		tempField = m_SegmentationRegistrationFilter->GetOutput();
		tempField->DisconnectPipeline();

		// Increment level counter.  
		m_CurrentLevel++;
		fixedLevel  = vnl_math_min((int)m_CurrentLevel, (int)GetNthFixedImagePyramid(1)->GetNumberOfLevels());
		movingLevel = vnl_math_min((int)m_CurrentLevel, (int)GetNthMovingImagePyramid(1)->GetNumberOfLevels());
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
		weightLevel = vnl_math_min((int)m_CurrentLevel, (int)GetNthWeightImagePyramid(1)->GetNumberOfLevels());
#endif
#ifdef USE_PROB_PRIOR
		probLevel   = vnl_math_min((int)m_CurrentLevel, (int)GetNthProbImagePyramid(1)->GetNumberOfLevels());
#endif

		// Invoke an iteration event.
		this->InvokeEvent(IterationEvent());

		// We can release data from pyramid which are no longer required.
		if (movingLevel > 0) {
			for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
				GetNthMovingImagePyramid(i)->GetOutput(movingLevel-1)->ReleaseData();
			}
		}
		if (fixedLevel > 0) {
			for (unsigned int i = 1; i <= NumberOfFixedChannels; i++) {
				GetNthFixedImagePyramid(i)->GetOutput(fixedLevel-1)->ReleaseData();
			}
		}
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
		if (weightLevel > 0) {
			for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
				GetNthWeightImagePyramid(i)->GetOutput(weightLevel-1)->ReleaseData();
			}
		}
#endif
#ifdef USE_PROB_PRIOR
		if (probLevel > 0) {
			for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
				GetNthProbImagePyramid(i)->GetOutput(probLevel-1)->ReleaseData();
			}
		}
#endif
	} // while not Halt()

	if (!lastShrinkFactorsAllOnes) {
		// Some of the last shrink factors are not one
		// graft the output of the expander filter to
		// to output of this filter

		// resample the field to the same size as the fixed image
		m_FieldExpander->SetInput(tempField);
		m_FieldExpander->SetSize(GetNthFixedImage(1)->GetLargestPossibleRegion().GetSize());
		m_FieldExpander->SetOutputStartIndex(GetNthFixedImage(1)->GetLargestPossibleRegion().GetIndex());
		m_FieldExpander->SetOutputOrigin(GetNthFixedImage(1)->GetOrigin());
		m_FieldExpander->SetOutputSpacing(GetNthFixedImage(1)->GetSpacing());
		m_FieldExpander->SetOutputDirection(GetNthFixedImage(1)->GetDirection());

		m_FieldExpander->UpdateLargestPossibleRegion();
		this->GraftOutput(m_FieldExpander->GetOutput());
	} else {
		// all the last shrink factors are all ones
		// graft the output of registration filter to
		// to output of this filter
		this->GraftOutput(tempField);
	}

	// Release memory
	m_FieldExpander->SetInput(NULL);
	m_FieldExpander->GetOutput()->ReleaseData();
	m_SegmentationRegistrationFilter->SetInput(NULL);
	m_SegmentationRegistrationFilter->GetOutput()->ReleaseData();
}

template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::StopRegistration()
{
	m_SegmentationRegistrationFilter->StopRegistration();
	m_StopRegistrationFlag = true;
}

template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
bool
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::Halt()
{
	// Halt the registration after the user-specified number of levels
	if (m_NumberOfLevels != 0) {
		this->UpdateProgress(static_cast<float>(m_CurrentLevel) / static_cast<float>(m_NumberOfLevels));
	}
	if (m_CurrentLevel >= m_NumberOfLevels) {
		return true;
	}
	if (m_StopRegistrationFlag) {
		return true;
	} else { 
		return false; 
	}
}

template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GenerateOutputInformation()
{
	typename DataObject::Pointer output;

	if (this->GetInput(0)) {
		// Initial deformation field is set.
		// Copy information from initial field.
		this->Superclass::GenerateOutputInformation();
	} else if (GetNthFixedImage(1)) {
		// Initial deforamtion field is not set. 
		// Copy information from the fixed image.
		for (unsigned int idx = 0; idx < this->GetNumberOfOutputs(); ++idx) {
			output = this->GetOutput(idx);
			if (output) {
				output->CopyInformation(GetNthFixedImage(1));
			}  
		}
	}
}

template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::GenerateInputRequestedRegion()
{
	// call the superclass's implementation
	Superclass::GenerateInputRequestedRegion();
	
	// request the largest possible region for the moving image
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		MovingImagePointer movingPtr = const_cast< MovingImageType * >(GetNthMovingImage(i));
		if (movingPtr) {
			movingPtr->SetRequestedRegionToLargestPossibleRegion();
		}
	}

	// just propagate up the output requested region for
	// the fixed image and initial deformation field.
	DeformationFieldPointer inputPtr = const_cast< DeformationFieldType * >(this->GetInput());
	DeformationFieldPointer outputPtr = this->GetOutput();
	if (inputPtr) {
		inputPtr->SetRequestedRegion(outputPtr->GetRequestedRegion());
	}

	for (unsigned int i = 1; i <= NumberOfFixedChannels; i++) {
		FixedImagePointer fixedPtr = const_cast< FixedImageType * >(GetNthFixedImage(i));
		if (fixedPtr) {
			fixedPtr->SetRequestedRegion(outputPtr->GetRequestedRegion());
		}
	}

#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		WeightImagePointer weightPtr = const_cast< WeightImageType * >(GetNthWeightImage(i));
		if (weightPtr) {
			weightPtr->SetRequestedRegion(outputPtr->GetRequestedRegion());
		}
	}
#endif

#ifdef USE_PROB_PRIOR
	for (unsigned int i = 1; i <= NumberOfMovingChannels; i++) {
		ProbImagePointer probPtr = const_cast< ProbImageType * >(GetNthProbImage(i));
		if (probPtr) {
			probPtr->SetRequestedRegion(outputPtr->GetRequestedRegion());
		}
	}
#endif
}

template <class TFixedImage, class TMovingImage, class TDeformationField, class TRealType, unsigned int VNumberOfFixedChannels, unsigned int VNumberOfMovingChannels>
void
MultiResolutionJointSegmentationRegistration<TFixedImage,TMovingImage,TDeformationField,TRealType,VNumberOfFixedChannels,VNumberOfMovingChannels>
::EnlargeOutputRequestedRegion(DataObject * ptr)
{
	// call the superclass's implementation
	Superclass::EnlargeOutputRequestedRegion(ptr);

	// set the output requested region to largest possible.
	DeformationFieldType * outputPtr;
	outputPtr = dynamic_cast<DeformationFieldType*>(ptr);

	if (outputPtr) {
		outputPtr->SetRequestedRegionToLargestPossibleRegion();
	}
}


} // end namespace itk


#endif
