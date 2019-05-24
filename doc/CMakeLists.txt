
set(DOCUMENTATION_NAME "GLISTR_Software_Manual" CACHE STRING "Documentation name")
mark_as_advanced(DOCUMENTATION_NAME)

if(BUILD_DOCUMENTATION)
	if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src)
		add_subdirectory(src)
	endif()
endif(BUILD_DOCUMENTATION)

install(FILES ${DOCUMENTATION_NAME}.pdf DESTINATION doc OPTIONAL)
