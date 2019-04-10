
add_subdirectory(Util)

set(GLISTR_SOURCES stdafx.cpp stdafx.h MyUtils.cpp MyUtils.h DMatrix.cpp DMatrix.h DVector.cpp DVector.h SMatrix.cpp SMatrix.h fams.cpp fams.h VolRegOpt.cpp VolRegOpt.h VolRegOptITK.cpp VolumeBP.cpp VolumeBP.h Volume.h FFD_table.h ENT_table.h itkJointSegmentationRegistrationFilter.h itkJointSegmentationRegistrationFilter.txx itkJointSegmentationRegistrationFunction.h itkJointSegmentationRegistrationFunction.txx itkMultiResolutionJointSegmentationRegistration.h itkMultiResolutionJointSegmentationRegistration.txx BrainTumorSegmentation.cpp)

#if(CUDA_FOUND)
#cuda_add_executable(GLISTR ${GLISTR_SOURCES} VolumeBP.cu)
#else(CUDA_FOUND)
add_executable(GLISTR ${GLISTR_SOURCES})
#endif(CUDA_FOUND)
add_executable(EvaluateQ stdafx.cpp stdafx.h MyUtils.cpp MyUtils.h Volume.h EvaluateQ.cpp)

if ( CMAKE_COMPILER_IS_GNUCC )
    # This is tested
    set_property( TARGET GLISTR APPEND_STRING PROPERTY COMPILE_FLAGS -w )
    set_property( TARGET EvaluateQ APPEND_STRING PROPERTY COMPILE_FLAGS -w )
endif ( CMAKE_COMPILER_IS_GNUCC )

if(ITKHasMetaIO EQUAL -1)
	set(GLISTR_LIBRARIES MetaIO NIFTI zlib SuperLU_MT ${ITK_LIBRARIES} ${LAPACK_LIBRARIES})
	set(EvaluateQ_LIBRARIES MetaIO NIFTI zlib ${ITK_LIBRARIES})
else()
	set(GLISTR_LIBRARIES NIFTI zlib SuperLU_MT ${ITK_LIBRARIES} ${LAPACK_LIBRARIES})
	set(EvaluateQ_LIBRARIES NIFTI zlib ${ITK_LIBRARIES})
endif()

if(WIN32)
	target_link_libraries(GLISTR ${GLISTR_LIBRARIES})
elseif(APPLE)	
	target_link_libraries(GLISTR ${GLISTR_LIBRARIES})
else()
	target_link_libraries(GLISTR ${GLISTR_LIBRARIES} -lrt)
endif()
target_link_libraries(EvaluateQ ${EvaluateQ_LIBRARIES})

set_target_properties(GLISTR PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(EvaluateQ PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS GLISTR DESTINATION bin)
install(TARGETS EvaluateQ DESTINATION bin)

if (UNIX)
  SET( original_exe   ${CMAKE_INSTALL_PREFIX}/bin/GLISTR)
  SET( linux_exe      ${CMAKE_INSTALL_PREFIX}/bin/glistr)
  ADD_CUSTOM_COMMAND( OUTPUT ${linux_exe}
                      COMMAND ln -s ${original_exe} ${linux_exe}
                      DEPENDS install ${original_exe} 
                      COMMENT "Generating legacy symbolic link" )
  #ADD_CUSTOM_TARGET( install_legacy DEPENDS ${original_exe} )
endif (UNIX)