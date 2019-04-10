
# find sphinx
find_program(SPHINX_EXECUTABLE NAMES sphinx-build
    HINTS
    $ENV{SPHINX_DIR}
    PATH_SUFFIXES bin
    DOC "Sphinx documentation generator"
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Sphinx DEFAULT_MSG
    SPHINX_EXECUTABLE
)

mark_as_advanced(SPHINX_EXECUTABLE)

# find latex
find_package(LATEX)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/conf.py"
    @ONLY)

add_custom_target(Documentation
    ${SPHINX_EXECUTABLE}
        -a -N -n -q -b html
        -c "${CMAKE_CURRENT_BINARY_DIR}"
        -d "${CMAKE_CURRENT_BINARY_DIR}/doctrees"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/html"
    COMMENT "Building HTML documentation with Sphinx")

add_custom_command(TARGET Documentation
    PRE_BUILD
    COMMAND ${SPHINX_EXECUTABLE}
        -a -N -n -q -b latex
        -c "${CMAKE_CURRENT_BINARY_DIR}"
        -d "${CMAKE_CURRENT_BINARY_DIR}/doctrees"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/latex" 
	COMMAND cd "${CMAKE_CURRENT_BINARY_DIR}/latex/" 
		&& ${PDFLATEX_COMPILER} "${DOCUMENTATION_NAME}.tex"
		&& ${PDFLATEX_COMPILER} "${DOCUMENTATION_NAME}.tex"
		&& ${PDFLATEX_COMPILER} "${DOCUMENTATION_NAME}.tex"
		&& ${MAKEINDEX_COMPILER} -s python.ist "${DOCUMENTATION_NAME}.idx"
		&& ${PDFLATEX_COMPILER} "${DOCUMENTATION_NAME}.tex"
		&& ${PDFLATEX_COMPILER} "${DOCUMENTATION_NAME}.tex"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/latex/${DOCUMENTATION_NAME}.pdf" "${PROJECT_SOURCE_DIR}/doc/src"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/latex/${DOCUMENTATION_NAME}.pdf" "${PROJECT_SOURCE_DIR}/doc"
    COMMENT "Building Latex documentation with Sphinx")




